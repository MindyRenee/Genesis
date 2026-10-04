"""Brain wave states — a coupled amplitude-phase oscillator.

## Modeling note

This is a **synthetic oscillator model** inspired by EEG, not an EEG
physiology model. Real EEG exhibits region-specific phase relationships
across cortical sources, absolute (not normalized) spectral power
measured in microvolts, and proper phase-amplitude coupling computed
via the Hilbert transform on band-passed signals. The oscillator here
uses normalized amplitudes and a single global phase per band, which
abstracts away spatial topology entirely. The claim that high gamma
and high phase-locking value (PLV) correspond to "cognitive access"
is a heuristic gating rule inspired by the global neuronal workspace
hypothesis, not a validated cognition marker. The citations
(Canolty & Knight, 2010) refer to the empirical phenomena that
motivated the design, not to claims that this oscillator reproduces
them.

Real brains produce rhythmic electrical patterns (brain waves) that
reflect different cognitive modes. These aren't just labels — they
gate how information flows:

- **Lambda (100-200 Hz)** — high-frequency oscillations (HFOs).
  Fast ripples and ripples associated with sharp wave-ripples
  during memory consolidation and seizure onset zones.
- **Gamma (30-80 Hz)** — information integration, binding disparate
  concepts into a coherent moment. When you "get it," that's gamma.
- **High Beta (20-30 Hz)** — hypervigilance, stress, anxiety. The
  "racing mind" of alertness that can't settle.
- **Mid Beta (15-20 Hz)** — active, alert thinking. Problem-solving,
  focused attention, motor decisions.
- **Low Beta (12-15 Hz)** — calm focus, sensorimotor rhythm (SMR).
  Quiet, introverted concentration with body stillness.
- **Sigma (12-16 Hz)** — sleep spindles during NREM sleep. Bursts of
  oscillatory activity that mediate memory consolidation.
- **Alpha (8-12 Hz)** — relaxed reflection. Alpha *inhibits* irrelevant
  information, filtering noise so the signal stands out. It's not
  just "calm" — it's active suppression of distraction.
- **Theta (4-8 Hz)** — memory consolidation, emotional processing,
  creativity. REM sleep is theta-dominant. Hippocampal replay happens
  here.
- **Delta (0.5-4 Hz)** — deep restorative sleep. Housekeeping,
  glymphatic clearance, noncognitive maintenance.
- **Epsilon (0.1-0.5 Hz)** — infraslow oscillations (ISF). Slow
  cortical potentials that modulate cortical excitability and
  coordinate large-scale network dynamics.

# How Genesis's brain waves work

Genesis's brain waves are produced by a **coupled amplitude-phase
oscillator** (``BrainWaveOscillator``), not a classifier. Each band
has persistent phase and amplitude state:

- **Phase** advances at the band's characteristic frequency (delta
  ~2.25 Hz, theta ~5.5 Hz, alpha ~10 Hz, sigma ~14 Hz, low beta
  ~13.5 Hz, mid beta ~17.5 Hz, high beta ~25 Hz, gamma ~50 Hz).
  Phase is genuine oscillator state — it precesses continuously,
  producing real periodicity.
- **Amplitude** relaxes toward neurochemically-driven targets with
  band-specific time constants. Slow bands have inertia (delta τ=8s);
  fast bands track quickly (gamma τ=0.5s). This produces temporal
  dynamics: brain wave state doesn't jump instantly when
  neurochemistry changes, it transitions.
- **Cross-frequency coupling** emerges from phase-amplitude
  interaction: the slow band's phase modulates the fast band's
  target amplitude (phase-amplitude coupling, the canonical mechanism
  for cross-scale neural coordination; Canolty & Knight, 2010).
- **Top-down drive** from cognition, action, and perception boosts
  specific bands above their neurochemical targets. Attention →
  gamma, memory retrieval → theta, motor planning → mid beta, V1
  visual input → gamma. This is bidirectional coupling: the oscillator
  gates cognition, and cognition drives the oscillator.

The neurochemical-to-target mapping is grounded in neuroscience:
- High alertness + high plasticity → gamma target (integration)
- High alertness + low plasticity → mid/high beta target (alert, not integrating)
- Moderate alertness → alpha target (reflective filtering)
- Low alertness → theta target (memory consolidation)
- Sleep phase → delta/theta/sigma target (restorative/dreaming/spindles)

# Drowsiness, REM, and regional topology

- **Drowsy is the alpha→theta descent.** Alpha drops out over
  occipital cortex as theta rises on the frontocentral midline
  (Hori A3/B1). It is not a fixed band ratio but a continuous
  function of arousal: at ~0.50 alpha and theta interleave, at
  ~0.30 theta dominates with delta creeping in (N1), and only
  below ~0.25 is delta taking over — at that point she is in
  slow-wave sleep, not drowsiness.
- **REM runs at waking metabolic intensity.** Theta is the
  predominant background, carried by its characteristic sawtooth
  carrier, concentrated in the hippocampus. Beta and gamma fire
  at the same intensity as waking cognitive work (REM has the
  highest brain-wide energy expenditure; Bergel et al., 2021).
  Alpha is not absent but intermittent — it flashes on the
  infraslow (epsilon) envelope, echoing quiet wakefulness rather
  than sustained relaxed-wake alpha.
- **Topology.** Each band's power is distributed over regions
  by physiology-weighted affinities (theta→hippocampus/temporal,
  alpha→occipital, delta→frontal in N3, sigma→central,
  gamma→prefrontal/parietal, SMR→central). The per-region
  distribution travels on ``BrainWaveState.regional_powers``.
- **Phase/wave coherence.** A waking summary whose arousal has
  fallen into the N1 band resolves to the drowsy signature —
  the same waves the explicit drowsy phase produces, so phase,
  waves, emotion, and learning cannot disagree.

# Cognitive effects

The oscillator's amplitude distribution produces three cognitive
gating parameters:

- `focus` — how narrow the attention (0=broad, 1=narrow)
- `integration` — how likely distant concepts connect (0=isolated, 1=bound)
- `consolidation` — how much memory consolidation happens (0=none, 1=maximal)

These gate how the cognition engine processes information:
- **Gamma**: boosts cross-network integration. Distant concepts are
  more likely to connect. Reasoning chains go deeper.
- **High Beta**: hypervigilant processing. Alert but rigid, can't
  see the big picture. Stress and anxiety.
- **Mid Beta**: standard active processing. Focused problem-solving.
- **Low Beta**: calm, focused attention. Sensorimotor rhythm —
  quiet concentration with body stillness.
- **Alpha**: filters peripheral concepts. Only highly-activated
  concepts reach cognition. Reduces noise, increases signal.
- **Theta**: prioritizes memory consolidation and emotional
  processing. Novel connections more likely. Creativity boosted.
- **Delta**: minimal cognitive processing. Housekeeping mode.
- **Sigma**: sleep spindles. Memory consolidation during NREM sleep.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, ClassVar

from genesis_client import NeuroSummary
from genesis_client.protocol import (
    DT_MAX,
    PHASE_ACTIVE,
    PHASE_ALERT,
    PHASE_DROWSY,
)

__all__ = [
    "N1_THETA_ALERTNESS",
    "BrainWave",
    "BrainWaveOscillator",
    "BrainWaveState",
    "CrossFrequencyCoupling",
    "GammaSynchrony",
    "SleepStage",
    "SleepStageSignature",
    "ThetaGammaCoupling",
    "add_brain_wave_drive",
    "add_stimulus",
    "apply_self_priority",
    "assess_brain_waves",
    "compute_gamma_synchrony",
    "compute_theta_gamma_coupling",
    "derive_self_priority",
    "meditative_coherence",
    "reset_oscillator",
    "resolve_waking_drowsy",
]


#: Alertness at or below which the waking descent has passed drowsy
#: onset (alpha) into the N1-like theta band. Drowsiness descends
#: alpha → theta → delta as alertness falls; a generic waking phase
#: read here is pre-sleep, so it resolves as drowsy — theta implies
#: N1, never active.
N1_THETA_ALERTNESS: float = 0.35


def resolve_waking_drowsy(phase: int, arousal: float) -> int:
    """Resolve a waking phase against arousal (pure; unit-testable).

    The daemon reports Drowsy only once sleep pressure (adenosine)
    joins low arousal, but the wave model reads drowsiness from
    arousal alone. A generic waking label (active/alert — the two
    phases rendered through the alertness-driven wave branch) with
    arousal inside the N1-like theta band is pre-sleep whatever
    adenosine says: return PHASE_DROWSY so phase, waves, emotion,
    and learning agree. Anything else passes through unchanged.
    """
    if phase in (PHASE_ACTIVE, PHASE_ALERT) and arousal <= N1_THETA_ALERTNESS:
        return PHASE_DROWSY
    return phase


class BrainWave(Enum):
    """The major brain wave bands."""

    LAMBDA = "lambda"    # 100-200 Hz — high-frequency oscillations (HFOs)
    GAMMA = "gamma"      # 30-80 Hz — integration, binding
    BETA3 = "beta3"      # 20-30 Hz — hypervigilance, stress
    BETA2 = "beta2"      # 15-20 Hz — active thinking, alert
    BETA1 = "beta1"      # 12-15 Hz — calm focus, SMR
    SIGMA = "sigma"      # 12-16 Hz — sleep spindles (NREM)
    ALPHA = "alpha"      # 8-12 Hz — relaxed reflection, filtering
    THETA = "theta"      # 4-8 Hz — memory, creativity, REM
    DELTA = "delta"      # 0.5-4 Hz — deep sleep, restoration
    EPSILON = "epsilon"  # 0.1-0.5 Hz — infraslow oscillations (ISF)


@dataclass(slots=True)
class BrainWaveState:
    """The current brain wave state from the coupled oscillator.

    Carries cognitive parameters that gate how the cognition engine
    processes information. The ``powers`` reflect the oscillator's
    normalised amplitudes; ``phases`` carry the instantaneous phase
    of each band (radians, [0, 2π)) — the raw oscillator state from
    which cross-frequency coupling is computed.
    """

    dominant: BrainWave
    secondary: BrainWave  # the next most active band

    # Power distribution (0-1 for each band, sums to ~1.0)
    powers: dict[BrainWave, float]

    # Cognitive gating parameters (derived from the wave distribution)
    focus: float  # 0=broad attention, 1=narrow focus
    integration: float  # 0=isolated concepts, 1=distant concepts bind
    consolidation: float  # 0=no memory consolidation, 1=maximal

    # Human-readable description
    label: str
    description: str

    # Cross-frequency phase-amplitude couplings (slow→fast pairs).
    # Each pair is keyed as "slow_band->fast_band" and carries the
    # phase-amplitude coupling strength and inferred cognitive mode.
    cross_frequency: dict[str, Any] = field(default_factory=dict)

    # Instantaneous phase of each band (radians, [0, 2π)).
    # Populated by the oscillator; used for top-down coupling and
    # phase-aware cross-frequency computation.
    phases: dict[BrainWave, float] = field(default_factory=dict)

    # Per-region power distribution — each region's normalised
    # band-power vector, derived from the global powers weighted by
    # band→region affinities (phase-aware). This is what makes the
    # model topographically correct: alpha over occipital cortex,
    # theta in the hippocampus during REM, frontal slow waves in
    # N3, spindles centrally, gamma over prefrontal/parietal cortex.
    regional_powers: dict[str, dict[BrainWave, float]] = field(
        default_factory=dict
    )

    def describe(self) -> str:
        """First-person description of the current brain wave state."""
        return f"mind in {self.label} — {self.description}"

    def describe_cognition(self) -> str:
        """How the brain wave state affects cognition."""
        parts = []
        if self.focus > 0.7:
            parts.append("narrowly focused")
        elif self.focus < 0.3:
            parts.append("broadly attentive")
        else:
            parts.append("moderately focused")

        if self.integration > 0.7:
            parts.append("making distant connections")
        elif self.integration < 0.3:
            parts.append("keeping concepts separate")

        if self.consolidation > 0.6:
            parts.append("consolidating memories")

        return ", ".join(parts)


# ─── Sleep stage (NREM-REM cycle) ──────────────────────────────


class SleepStage(Enum):
    """The sleep stages of the NREM-REM cycle, derived from arousal.

    During the emergent "sleeping" phase, arousal level distinguishes
    the stages (AASM scoring; Iber et al., 2007). This mirrors the
    finer-grained staging used by the inner-life sleep system and is
    kept here (rather than imported) so ``brain_waves`` stays
    decoupled from the cognitive-mind package.

    For the legacy "sleeping" phase (where NREM/REM weren't
    distinguished), arousal > 0.50 maps to REM. For the explicit
    "nrem" phase, only N1/N2/N3 are produced:

        0.35 < arousal         → N1
        0.20 < arousal ≤ 0.35  → N2
        arousal ≤ 0.20         → N3
    """

    REM = "rem"
    N1 = "n1"
    N2 = "n2"
    N3 = "n3"

    @property
    def is_rem(self) -> bool:
        """Whether this stage is REM sleep."""
        return self is SleepStage.REM

    @property
    def is_nrem(self) -> bool:
        """Whether this stage is a NREM stage (N1, N2, or N3)."""
        return self is not SleepStage.REM


def _sleep_stage_from_summary(summary: NeuroSummary) -> SleepStage | None:
    """Derive the sleep stage from a neurochemical summary.

    Returns None if the summary is not in a sleep phase. Handles both
    the legacy "sleeping" phase name and the split "nrem"/"rem" phase
    names (the Rust side now distinguishes NREM from REM).

    For the legacy "sleeping" phase, arousal > 0.50 maps to REM
    (since the old daemon didn't distinguish NREM from REM). For the
    explicit "nrem" phase, only N1/N2/N3 are produced — the daemon
    said NREM, so it's NREM regardless of arousal.
    """
    phase = summary.phase_name
    if phase == "rem":
        return SleepStage.REM
    if phase == "sleeping":
        # Legacy single sleep phase — arousal distinguishes REM from NREM.
        arousal = summary.arousal
        if arousal > 0.50:
            return SleepStage.REM
        elif arousal > 0.35:
            return SleepStage.N1
        elif arousal > 0.20:
            return SleepStage.N2
        else:
            return SleepStage.N3
    if phase == "nrem":
        # Explicit NREM — stage depends on arousal, but never REM.
        arousal = summary.arousal
        if arousal > 0.35:
            return SleepStage.N1
        elif arousal > 0.20:
            return SleepStage.N2
        else:
            return SleepStage.N3
    return None


@dataclass(slots=True)
class SleepStageSignature:
    """The brain-wave signature of a specific sleep stage.

    Each sleep stage has a characteristic EEG signature
    (Iber et al., 2007; Steriade, 2003):

    - **REM**: theta-dominant (4-8 Hz) with beta activity and PGO
      waves; resembles waking EEG ("paradoxical sleep") but with
      muscle atonia. Drives emotional memory processing.
    - **N1**: mixed theta/alpha, low voltage; the transition into
      sleep. Slow eye movements.
    - **N2**: theta/delta background punctuated by sleep spindles
      (12-16 Hz sigma bursts) and K-complexes — the defining
      graphoelements of N2.
    - **N3**: delta-dominant (0.5-4 Hz) slow waves — slow-wave sleep
      (SWS). Sharpest wave-ripples, glymphatic clearance.
    """

    stage: SleepStage
    dominant: BrainWave
    secondary: BrainWave
    # Stage-specific graphoelements present in this stage
    features: list[str] = field(default_factory=list)
    # Human-readable description
    description: str = ""

    def describe(self) -> str:
        """First-person description of the sleep-stage signature."""
        feats = ", ".join(self.features) if self.features else "none"
        return (
            f"{self.stage.value.upper()} signature: "
            f"{self.dominant.value}-dominant with {feats} — {self.description}"
        )


# How close a computed cycle count must be to a whole number of turns
# before it is treated as exactly whole (see
# BrainWaveOscillator._advance_phase_exact).
#
# The float64 spacing of a cycle count is 2^-52 * magnitude: ~2.4e-15
# turns at 11 turns (THETA over 2 s), ~6.2e-15 at 28 (SIGMA over 2 s),
# and ~3.3e-13 at the largest count reachable within DT_MAX (lambda,
# 150 Hz over 10 s). 1e-9 turns is three to six orders of magnitude
# above all of those, while being 6.3e-9 rad -- far below anything the
# downstream phase consumers can observe. A genuinely fractional cycle
# count (a band whose frequency times the interval is not near-integral,
# e.g. alpha at 10 Hz over 0.2 s = 2.0 exactly, or 5.5 Hz over 0.2 s =
# 1.1) sits 0.1 turns or more away, so it is never snapped.
_TURN_SNAP_EPS = 1e-9


# ─── Theta-gamma cross-frequency coupling ──────────────────────


@dataclass(slots=True)
class ThetaGammaCoupling:
    """Theta-gamma cross-frequency coupling (CFC) state.

    Gamma amplitude (~30-80 Hz) is modulated by the phase of the
    theta oscillation (~4-8 Hz). This phase-amplitude coupling (PAC)
    is one of the most robust cross-frequency relationships in the
    brain and is critical for memory:

    - **Encoding**: gamma power peaks near the theta *peak*,
      when hippocampal pyramidal cells are most
      excitable — favouring the formation of new associations
      (Hasselmo et al., 2002; Douchamps et al., 2013;
      Newman et al., 2013; di Chanaz et al., 2023).
    - **Retrieval**: gamma power peaks near the theta *trough*,
      favouring the reactivation of stored
      representations (Hasselmo et al., 2002; Douchamps et al., 2013).

    ``coupling_strength`` (0-1) measures how strongly gamma amplitude
    is locked to theta phase — high coupling means gamma bursts are
    reliably timed to a particular theta phase, which is the neural
    substrate of organised memory processing. Reduced theta-gamma
    coupling is observed in cognitive ageing and schizophrenia
    (Tort et al., 2008; Lisman & Jensen, 2013).
    """

    theta_power: float  # 0-1, theta band power
    gamma_power: float  # 0-1, gamma band power
    # Preferred theta phase (radians) at which gamma amplitude peaks.
    # ~0 = theta peak (encoding-favouring); ~π = theta trough
    # (retrieval-favouring).
    preferred_phase: float = 0.0
    # Coupling strength: 0 = no phase-amplitude coupling,
    # 1 = gamma perfectly locked to theta phase.
    coupling_strength: float = 0.0
    # Dominant cognitive mode implied by the preferred phase.
    mode: str = "balanced"  # "encoding", "retrieval", or "balanced"

    def describe(self) -> str:
        """Human-readable description of the coupling."""
        return (
            f"[theta-gamma CFC] strength {self.coupling_strength:.2f}, "
            f"preferred phase {self.preferred_phase:.2f} rad ({self.mode})"
        )


@dataclass(slots=True)
class CrossFrequencyCoupling:
    """Generic slow→fast phase-amplitude coupling.

    In real brains, the phase of a slow oscillation (delta, theta,
    alpha) gates the amplitude of a faster one (alpha, beta, gamma).
    This is the canonical mechanism for cross-scale coordination:
    slow rhythms set the temporal windows in which fast, local
    computations can occur (Canolty & Knight, 2010).
    """

    slow_band: str
    fast_band: str
    coupling_strength: float  # 0-1, phase-amplitude modulation strength
    mode: str  # "gating", "integration", "maintenance", "rest", "idle"

    def describe(self) -> str:
        """Human-readable description of the coupling."""
        return (
            f"[{self.slow_band}→{self.fast_band} CFC] "
            f"strength {self.coupling_strength:.2f} ({self.mode})"
        )


def compute_theta_gamma_coupling(
    state: BrainWaveState,
    consolidation_weight: float = 0.5,
    encoding_weight: float = 0.5,
) -> ThetaGammaCoupling:
    """Compute theta-gamma cross-frequency coupling from brain wave state.

    Gamma amplitude is modulated by theta phase. The coupling strength
    depends on how much *both* theta and gamma power are present —
    coupling is only meaningful when there is a carrier (theta) to
    modulate and a signal (gamma) to be modulated. We model the
    modulation index as the geometric mean of the two powers, scaled
    by the neurochemical gating (consolidation/encoding weights drive
    the theta/gamma balance).

    The preferred phase — and thus the encoding vs retrieval mode —
    determined by the encoding/consolidation balance: high encoding
    weight biases toward the theta peak (encoding), high
    consolidation weight biases toward the theta trough (retrieval/
    consolidation replay).

    Args:
        state: The current brain wave state (from ``assess_brain_waves``).
        consolidation_weight: Neurochemical consolidation weight [0,1].
        encoding_weight: Neurochemical encoding weight [0,1].

    Returns:
        A ``ThetaGammaCoupling`` describing the phase-amplitude
        coupling.
    """
    theta = state.powers[BrainWave.THETA]
    gamma = state.powers[BrainWave.GAMMA]

    # Coupling strength: geometric mean of theta and gamma power,
    # scaled by how balanced the neuromodulatory gating is. When
    # either band is absent, coupling collapses — matching the
    # biology that PAC requires both a carrier and a signal.
    raw = math.sqrt(theta * gamma)
    # Gate by the combined memory-processing weight
    gate = 0.5 + 0.5 * (consolidation_weight + encoding_weight) / 2.0
    coupling = raw * gate
    coupling = max(0.0, min(1.0, coupling))

    # Preferred phase: encoding bias → theta peak (0, encoding);
    # consolidation bias → theta trough (π, retrieval/replay).
    # Interpolate by the encoding-vs-consolidation balance.
    total = encoding_weight + consolidation_weight
    if total <= 0.0:
        balance = 0.5
    else:
        balance = encoding_weight / total  # 0 = all consolidation, 1 = all encoding
    # balance 0 → phase π (trough, retrieval); balance 1 → phase 0 (peak, encoding)
    preferred_phase = (1.0 - balance) * math.pi

    if balance > 0.6:
        mode = "encoding"
    elif balance < 0.4:
        mode = "retrieval"
    else:
        mode = "balanced"

    return ThetaGammaCoupling(
        theta_power=theta,
        gamma_power=gamma,
        preferred_phase=preferred_phase,
        coupling_strength=coupling,
        mode=mode,
    )


def _cognitive_mode_for_coupling(slow: str, fast: str) -> str:
    """Infer a cognitive mode from a slow→fast band pair.

    The mode is a qualitative label derived from the canonical
    electrophysiological roles of the two bands. It is not a hard-coded
    threshold; it follows the band names themselves.
    """
    # Fast band gives the *kind* of processing being gated
    fast_role = {
        "lambda": "fast_ripple",
        "gamma": "integration",
        "beta3": "hypervigilance",
        "beta2": "active_thinking",
        "beta1": "calm_focus",
        "sigma": "consolidation",
        "alpha": "gating",
    }.get(fast, "coupling")
    # Slow band gives the *state* in which the gating happens
    slow_role = {
        "epsilon": "infraslow_modulation",
        "delta": "rest",
        "theta": "encoding",
        "alpha": "filtering",
        "sigma": "consolidation",
    }.get(slow, slow)
    # Combine: e.g. theta-gamma → "encoding-integration"
    return f"{slow_role}-{fast_role}"


# ─── Gamma synchrony (40 Hz hypothesis) ────────────────────────


@dataclass(slots=True)
class GammaSynchrony:
    """Gamma-band synchrony across brain regions — the 40 Hz binding model.

    The 40 Hz hypothesis (Crick & Koch, 1990; Crick, 1994) proposes
    that gamma-band (~40 Hz) synchrony *binds* distributed neural
    representations into a unified cognitive percept. Dehaene's
    global neuronal workspace theory extends this: cognitive access
    occurs when gamma synchrony ignites across a distributed
    fronto-parietal network (Dehaene & Naccache, 2001; Dehaene,
    2014).

    - **High gamma synchrony** → distributed regions fire in phase,
      binding features into a coherent representation → cognitive
      access prediction.
    - **Low gamma synchrony** → regions fire independently →
      processing remains local/noncognitive (subliminal).

    ``synchrony`` (0-1) is the phase-locking across modelled regions.
    ``cognitive_access_prediction`` is the binary prediction derived
    from a threshold (Dehaene, 2014): synchrony above ~0.5 predicts
    cognitive access; below predicts noncognitive processing.
    """

    # Per-region gamma power (0-1). Keys are region names.
    region_powers: dict[str, float] = field(default_factory=dict)
    # Cross-region phase-locking value (PLV) in [0, 1]. 0 = no
    # synchrony, 1 = perfect phase locking across regions.
    synchrony: float = 0.0
    # Predicted cognitive access based on the 40 Hz hypothesis.
    cognitive_access_prediction: bool = False
    # Human-readable description
    description: str = ""

    def describe(self) -> str:
        """Human-readable description of the gamma synchrony state."""
        access = "cognitive access" if self.cognitive_access_prediction else "noncognitive"
        return (
            f"[gamma synchrony] PLV {self.synchrony:.2f} across "
            f"{len(self.region_powers)} regions → {access}"
        )


# Modelled cortical regions for gamma synchrony. These are the key
# nodes of the global neuronal workspace (Dehaene, 2014): prefrontal
# and parietal association areas, plus sensory regions that feed them.
_GAMMA_REGIONS = ("prefrontal", "parietal", "temporal", "occipital")


def compute_gamma_synchrony(
    state: BrainWaveState,
    plasticity: float = 0.5,
) -> GammaSynchrony:
    """Compute a gamma-synchrony proxy across brain regions.

    Gamma synchrony is modelled from the global gamma power (the
    "ignition" signal) gated by plasticity (which determines whether
    long-range cortico-cortical connections can sustain phase locking
    — chronic stress / low plasticity impairs gamma synchrony,
    matching the stress-gamma reduction in the neurobiology audit;
    Herrmann & Demiralp, 2012).

    Each region receives a gamma power drawn from the global gamma
    power with small region-specific variation. The synchrony index
    across regions is the global gamma power scaled by plasticity —
    high gamma + high plasticity → high synchrony → cognitive
    access; low gamma or low plasticity → low synchrony.

    Note: this is a heuristic proxy, NOT a true phase-locking value.
    A true PLV is PLV = |⟨exp(i·Δφ)⟩| over phase samples (Lachaux
    et al., 1999; Canolty & Knight, 2010) and has no universal 0.5
    threshold. Global-neuronal-workspace ignition (Dehaene, 2014) is a
    nonlinear dynamical transition, not a fixed PLV cutoff; the 0.5
    cutoff below is an operational heuristic for this synthetic model.

    Args:
        state: The current brain wave state.
        plasticity: Neurochemical plasticity gate [0,1].

    Returns:
        A ``GammaSynchrony`` with per-region powers, synchrony proxy,
        and a cognitive-access prediction.
    """
    gamma = state.powers[BrainWave.GAMMA]
    # Long-range phase locking requires both gamma drive and the
    # plasticity to sustain cortico-cortical coherence. Gamma power
    # here is a normalised fraction (sums to ~1 across bands), so we
    # scale it up so that genuinely high-gamma cognitive states
    # (flow, active+plastic) can cross the cognitive-access threshold,
    # while low-gamma states (sleep, stress) stay well below it.
    plv = max(0.0, min(1.0, gamma * (1.2 + plasticity * 0.8)))

    # Per-region gamma power: global gamma with a small deterministic
    # spread so regions aren't identical (fronto-parietal dominance
    # during cognitive access — Dehaene, 2014).
    region_powers: dict[str, float] = {}
    region_phases: dict[str, float] = {}
    for region in _GAMMA_REGIONS:
        # Frontal/parietal (workspace hubs) get a slight boost;
        # sensory regions slightly lower.
        hub_boost = 0.1 if region in ("prefrontal", "parietal") else -0.05
        power = max(0.0, min(1.0, gamma + hub_boost * gamma))
        region_powers[region] = power
        # Regional phase: small deterministic offset from global gamma phase
        # to simulate regional phase differences (Lachaux et al., 1999)
        phase_offset = (
            0.0 if region == "prefrontal"
            else 0.3 if region == "parietal"
            else 0.6 if region == "temporal"
            else 0.9
        )
        region_phases[region] = (
            state.phases.get(BrainWave.GAMMA, 0.0) + phase_offset
        ) % (2.0 * math.pi)

    # True PLV: |<exp(i*delta_phi)>| across region pairs (Lachaux et al., 1999)
    # PLV = |(1/N) * sum(exp(i*(phi_1 - phi_2)))|
    # For N regions, compute all pairwise phase differences
    regions = list(region_phases.keys())
    if len(regions) >= 2:
        plv_sum = 0.0
        plv_count = 0
        for i in range(len(regions)):
            for j in range(i + 1, len(regions)):
                delta_phi = region_phases[regions[i]] - region_phases[regions[j]]
                plv_sum += math.cos(delta_phi)  # Re(exp(i*delta_phi))
                plv_count += 1
        true_plv = abs(plv_sum / plv_count) if plv_count > 0 else 0.0
    else:
        true_plv = 0.0

    # Blend the heuristic PLV with the true PLV
    # The heuristic captures gamma power + plasticity gating
    # The true PLV captures phase coherence across regions
    plv = 0.5 * plv + 0.5 * true_plv

    # Cognitive-access heuristic: synchrony proxy > 0.5 predicts access.
    # Operational cutoff for this synthetic model only — not a
    # neuroscientific constant (see docstring).
    cognitive = plv > 0.5

    if cognitive:
        description = "gamma ignition across the global workspace"
    elif plv > 0.25:
        description = "partial gamma coherence, subliminal processing"
    else:
        description = "no gamma binding, local noncognitive processing"

    return GammaSynchrony(
        region_powers=region_powers,
        synchrony=plv,
        cognitive_access_prediction=cognitive,
        description=description,
    )


# ─── Regional (topographic) attribution ─────────────────────────
#
# Real scalp EEG is a topographic measurement: each band has a
# prominent generator site. Alpha (8–12 Hz) emerges over the
# occipital cortex with eyes closed. Theta is hippocampal/medial
# temporal, with frontal midline theta indexing cognitive effort
# (Cavanagh & Frank, 2014). Slow-wave delta peaks over frontal
# cortex (Huber et al., 2000). Sleep spindles (sigma) are maximal
# centrally. Beta is frontocentral (SMR centrally). Gamma that
# binds representations is prefrontal/parietal (Dehaene, 2014).
# REM theta is hippocampus-predominant (Buzsaki, 2002; Montgomery
# et al., 2008) and REM sawtooth waves are frontocentral (JNeurosci
# 40:8900). Lambda waves occur occipital during visual scanning.

_REGIONS: tuple[str, ...] = (
    "prefrontal",
    "frontal",
    "parietal",
    "temporal",
    "occipital",
    "central",
    "hippocampus",
)

# Band → region weight. Values are relative prominence of that band
# in that region. Phase-aware boosts (REM hippocampal theta, N3
# frontal delta) are applied on top in _regional_powers.
_BAND_REGION_AFFINITY: dict[BrainWave, dict[str, float]] = {
    BrainWave.EPSILON: {
        "prefrontal": 0.6, "frontal": 0.7, "parietal": 0.5,
        "temporal": 0.4, "occipital": 0.4, "central": 0.5,
        "hippocampus": 0.4,
    },
    BrainWave.DELTA: {
        "prefrontal": 0.8, "frontal": 1.0, "parietal": 0.6,
        "temporal": 0.5, "occipital": 0.5, "central": 0.7,
        "hippocampus": 0.4,
    },
    BrainWave.THETA: {
        "prefrontal": 0.7, "frontal": 0.6, "parietal": 0.5,
        "temporal": 0.9, "occipital": 0.3, "central": 0.5,
        "hippocampus": 1.0,
    },
    BrainWave.ALPHA: {
        "prefrontal": 0.3, "frontal": 0.4, "parietal": 0.7,
        "temporal": 0.6, "occipital": 1.0, "central": 0.4,
        "hippocampus": 0.2,
    },
    BrainWave.SIGMA: {
        "prefrontal": 0.4, "frontal": 0.5, "parietal": 0.8,
        "temporal": 0.4, "occipital": 0.4, "central": 1.0,
        "hippocampus": 0.2,
    },
    BrainWave.BETA1: {
        "prefrontal": 0.5, "frontal": 0.6, "parietal": 0.5,
        "temporal": 0.4, "occipital": 0.3, "central": 1.0,
        "hippocampus": 0.3,
    },
    BrainWave.BETA2: {
        "prefrontal": 0.8, "frontal": 0.9, "parietal": 0.6,
        "temporal": 0.5, "occipital": 0.4, "central": 0.7,
        "hippocampus": 0.3,
    },
    BrainWave.BETA3: {
        "prefrontal": 0.9, "frontal": 1.0, "parietal": 0.5,
        "temporal": 0.5, "occipital": 0.3, "central": 0.6,
        "hippocampus": 0.2,
    },
    BrainWave.GAMMA: {
        "prefrontal": 1.0, "frontal": 0.8, "parietal": 0.9,
        "temporal": 0.7, "occipital": 0.7, "central": 0.5,
        "hippocampus": 0.8,
    },
    BrainWave.LAMBDA: {
        "prefrontal": 0.2, "frontal": 0.2, "parietal": 0.3,
        "temporal": 0.4, "occipital": 1.0, "central": 0.2,
        "hippocampus": 0.2,
    },
}


def _regional_powers(
    powers: dict[BrainWave, float], phase_name: str
) -> dict[str, dict[BrainWave, float]]:
    """Distribute global band powers over cortical/subcortical regions.

    Each region's band weights are the product of global band power
    and the band→region affinity, normalised within the region, so
    every region carries a full distribution (sum to 1). Phase-aware
    boosts make the topology state-dependent: REM theta concentrates
    in the hippocampus, N3 slow waves concentrate frontally.
    """
    affinity = {band: dict(regions) for band, regions in _BAND_REGION_AFFINITY.items()}
    if phase_name == "rem":
        # REM: hippocampal theta predominates; theta-gamma coupling
        # is strongest in hippocampus/RSC during phasic REM
        # (Montgomery et al., 2008; Bergel et al., 2021).
        affinity[BrainWave.THETA]["hippocampus"] *= 2.0
        affinity[BrainWave.THETA]["temporal"] *= 1.2
        affinity[BrainWave.GAMMA]["hippocampus"] *= 1.5
    if phase_name in ("nrem", "sleeping") and powers[BrainWave.DELTA] >= 0.4:
        # N3 slow waves are frontally predominant (Huber et al., 2000).
        affinity[BrainWave.DELTA]["frontal"] *= 1.5
        affinity[BrainWave.DELTA]["prefrontal"] *= 1.2
    out: dict[str, dict[BrainWave, float]] = {}
    for region in _REGIONS:
        raw = {
            band: powers[band] * affinity[band][region]
            for band in BrainWave
        }
        total = sum(raw.values())
        if total > 0.0:
            out[region] = {band: value / total for band, value in raw.items()}
        else:
            out[region] = dict.fromkeys(BrainWave, 0.0)
    return out


def _sawtooth(phase: float) -> float:
    """Sawtooth carrier for REM theta (4–8 Hz).

    REM theta is sawtooth-shaped — a fast ramp with a sharp notch
    (Rodenbeck et al., 2006; Buzsaki, 2002). In contrast to the
    symmetric cosine used for non-REM bands, this asymmetry is part
    of what makes hippocampal REM theta distinct. Returns values in
    [-1, 1).
    """
    return (phase / math.pi) - 1.0


def _wave_phase_value(
    band: BrainWave, phase: float, phase_name: str
) -> float:
    """The phase-carrier value for a slow band at this moment.

    REM theta uses its characteristic sawtooth carrier; every other
    band uses the symmetric cosine. This asymmetry feeds the PAC
    modulation of fast bands, so gamma during REM is timed to the
    sawtooth ramp rather than a sine peak.
    """
    if phase_name == "rem" and band is BrainWave.THETA:
        return _sawtooth(phase)
    return math.cos(phase)


def _rem_alpha_envelope(epsilon_phase: float) -> float:
    """Intermittent REM alpha — infraslow (epsilon-band) burst gain.

    During REM, alpha rhythms are fragmented: brief alpha intrusions
    echo quiet-wakefulness/meditative alpha, waxing and waning with
    the infraslow (~0.25 Hz) envelope (periodic ~4 s). The envelope
    is ``max(0, cos(φ))²``: a flash reaching full gain near cosine
    peak, falling to zero at cosine trough — never the sustained
    alpha of eyes-closed wake.
    """
    return max(0.0, math.cos(epsilon_phase)) ** 2



def meditative_coherence(state: BrainWaveState) -> float:
    """How coherently entrained the wave state is to meditation bands.

    Meditative states are characterized by alpha dominance with a
    theta component — relaxed, internally directed attention (Cahn &
    Polich, 2006). Beta (task engagement) and gamma (binding effort)
    are anti-correlated; delta means sleep, not meditation, so it
    counts against coherence too.

    The metric is the share of total band power in the meditative
    bands (alpha + theta) — 1.0 when the spectrum is fully entrained
    to alpha/theta, ~0.4 for a typical awake beta-dominant mix, ~0
    in pure beta/gamma or delta states.

    Used by the harmonic retune during meditation: the more coherent
    her oscillatory state, the deeper the neurochemical convergence
    toward homeostatic set-points — a genuine two-way loop, since the
    retune itself calms the chemistry that drives the oscillator.
    """
    powers = state.powers
    total = sum(powers.values())
    if total <= 0.0:
        return 0.0
    calm = powers.get(BrainWave.ALPHA, 0.0) + powers.get(BrainWave.THETA, 0.0)
    return max(0.0, min(1.0, calm / total))

# ─── Coupled amplitude-phase oscillator ─────────────────────────
#
# The oscillator is the dynamical core. Each band has persistent
# phase (advancing at its characteristic frequency) and amplitude
# (relaxing toward neurochemically-driven targets). Cross-frequency
# coupling emerges from actual phase-amplitude interaction: the slow
# band's phase modulates the fast band's target amplitude. Top-down
# drive from cognition, action, and perception can boost specific
# bands above their neurochemical targets — bidirectional coupling.


class BrainWaveOscillator:
    """Coupled amplitude-phase oscillator for brain wave dynamics.

    Each band has persistent phase and amplitude state:

    - **Phase** advances at the band's characteristic frequency
      (delta ~2.25 Hz, theta ~5.5 Hz, alpha ~10 Hz, sigma ~14 Hz,
      low beta ~13.5 Hz, mid beta ~17.5 Hz, high beta ~25 Hz,
      gamma ~50 Hz). Phase is genuine oscillator state — it precesses
      continuously, producing real periodicity.
    - **Amplitude** relaxes toward neurochemically-driven targets
      with band-specific time constants. Slow bands relax slowly
      (delta τ=8s); fast bands relax quickly (gamma τ=0.5s). This
      produces temporal inertia: brain wave state doesn't jump
      instantly when neurochemistry changes, it transitions.
    - **Cross-frequency coupling** emerges from phase-amplitude
      interaction: the slow band's phase modulates the fast band's
      target amplitude. When theta is at its peak, gamma's target
      gets a boost; when theta is at its trough, gamma's target
      is suppressed. This is the canonical PAC mechanism.
    - **Top-down drive** from cognition/action/perception boosts
      specific bands above their neurochemical targets. Attention
      → gamma, memory retrieval → theta, motor activity → mid beta.
      The drive decays exponentially (τ=2s), so it must be
      sustained by ongoing activity.

    The oscillator is a coupled phase oscillator with amplitude
    dynamics — a Kuramoto-style population model where each band
    is an independent oscillator whose amplitude relaxes toward
    neurochemically-driven targets. It produces genuine
    oscillatory state (frequency, phase, amplitude) from which
    band powers, cross-frequency coupling, and cognitive gating
    parameters are derived.
    """

    # Characteristic frequencies (Hz) — center of each band
    _FREQS: ClassVar[dict[BrainWave, float]] = {
        BrainWave.EPSILON: 0.25,   # 0.1-0.5 Hz
        BrainWave.DELTA: 2.25,     # 0.5-4 Hz
        BrainWave.THETA: 5.5,      # 4-8 Hz
        BrainWave.ALPHA: 10.0,     # 8-12 Hz
        BrainWave.SIGMA: 14.0,     # 12-16 Hz
        BrainWave.BETA1: 13.5,     # 12-15 Hz
        BrainWave.BETA2: 17.5,     # 15-20 Hz
        BrainWave.BETA3: 25.0,     # 20-30 Hz
        BrainWave.GAMMA: 50.0,     # 30-80 Hz
        BrainWave.LAMBDA: 150.0,   # 100-200 Hz
    }

    # Amplitude relaxation time constants (seconds).
    # Slow bands have inertia; fast bands track quickly.
    _TAU: ClassVar[dict[BrainWave, float]] = {
        BrainWave.EPSILON: 12.0,
        BrainWave.DELTA: 8.0,
        BrainWave.THETA: 4.0,
        BrainWave.ALPHA: 2.0,
        BrainWave.SIGMA: 1.5,
        BrainWave.BETA1: 1.2,
        BrainWave.BETA2: 1.0,
        BrainWave.BETA3: 0.8,
        BrainWave.GAMMA: 0.5,
        BrainWave.LAMBDA: 0.3,
    }

    # Top-down drive decay time constant (seconds)
    _DRIVE_TAU: ClassVar[float] = 2.0

    # ── Integration resolution ────────────────────────────────────
    # Largest step the *relaxation* dynamics are integrated with.
    # Phase is exempt (see _advance_phase_exact); the amplitude
    # relaxation, PAC modulation, and drive/stimulus decays are
    # history-dependent, so they are sub-stepped to keep their
    # numerical behaviour independent of how often assess() is
    # called.
    #
    # 0.1 s is the resolution the neurochemical dynamics were
    # designed at (NeuroTickParams::DEFAULT.dt = 0.1 s). Matching it
    # means the wave and chemical integrators agree on what "one step"
    # means, which is what keeps the two clocks locked together.
    _MAX_SUBSTEP: ClassVar[float] = 0.1

    # Hard ceiling on sub-steps per assess() call, so a multi-second
    # stall cannot cost unbounded CPU. Past this the relaxation is
    # integrated once with the full dt: first-order exponential
    # relaxation is exact under composition, so fewer, larger steps
    # differ only by the PAC modulation sampled less often (see
    # _advance_relaxation).
    _MAX_SUBSTEPS: ClassVar[int] = 64

    # PAC modulation depth — how much slow phase modulates fast
    # amplitude target (fraction of fast band's base target)
    _PAC_DEPTH: ClassVar[float] = 0.15

    # Number of phase bins for modulation index computation (Tort et al., 2010)
    _PAC_BINS: ClassVar[int] = 18

    def __init__(self) -> None:
        """Initialize the brain wave oscillator."""
        self._phase: dict[BrainWave, float] = dict.fromkeys(BrainWave, 0.0)
        self._amplitude: dict[BrainWave, float] = dict.fromkeys(BrainWave, 0.2)
        self._top_down: dict[BrainWave, float] = dict.fromkeys(BrainWave, 0.0)
        self._stimulus: dict[BrainWave, float] = dict.fromkeys(BrainWave, 0.0)
        self._last_target_dominant: BrainWave | None = None
        self._last_phase_name: str | None = None
        self._last_sleep_stage: SleepStage | None = None
        self._last_time: float | None = None
        self._tick_count: int = 0

    def add_stimulus(
        self, band: BrainWave, strength: float, decay: float = 0.95
    ) -> None:
        """Apply continuous stimulus to a band.

        Inner (emotional, cognitive) and outer (sensory) stimulus
        continuously modulate band amplitudes. Unlike top-down drive
        (which is discrete and decays with τ=2s), stimulus is a
        persistent input that must be re-applied each cycle but
        decays slowly (default 0.95 per tick) to model the brain's
        continuous response to ongoing conditions.

        Args:
            band: The band to stimulate.
            strength: Stimulus strength, can be positive (excitation)
                or negative (inhibition). Typical range [-0.5, 0.5].
            decay: Per-tick decay factor (0-1). Higher = slower decay.
        """
        self._stimulus[band] = (
            self._stimulus[band] * decay + strength
        )

    def add_top_down_drive(
        self, band: BrainWave, strength: float
    ) -> None:
        """Add top-down drive to a band.

        Cognition, action, and perception can boost specific bands
        above their neurochemical targets. This is the bidirectional
        coupling that makes the oscillator a genuine dynamical system
        rather than a classifier reading neurochemistry.

        The drive decays exponentially (τ=2s), so it must be sustained
        by ongoing activity. A brief attention spike produces a brief
        gamma boost; sustained focus produces sustained gamma.

        Args:
            band: The band to drive.
            strength: Drive strength (0-1, added to band target).
        """
        self._top_down[band] = min(
            1.0, self._top_down[band] + max(0.0, strength)
        )

    def assess(
        self,
        summary: NeuroSummary,
        sleep_stage: SleepStage | None = None,
    ) -> BrainWaveState:
        """Advance the oscillator and return the current state.

        Computes neurochemically-driven target amplitudes, advances
        phase, relaxes amplitude toward targets (with cross-frequency
        phase-amplitude modulation), applies top-down drive and
        continuous stimulus, and builds a ``BrainWaveState`` from
        the resulting amplitudes.

        Args:
            summary: Current neurochemical summary.
            sleep_stage: Authoritative sleep stage from the ultradian
                tracker, when sleeping. This overrides the arousal-
                derived stage — see :func:`_sleep_phase_powers` for
                why arousal cannot supply it.

        The oscillator is dynamic: amplitudes flow continuously toward
        targets modulated by both inner (neurochemical) and outer
        (stimulus) inputs.
        """
        # Compute targets from neurochemistry (same classification
        # logic as before — these are the neurochemically-driven
        # equilibrium amplitudes)
        targets, label, description = self._compute_targets(
            summary, sleep_stage
        )

        # Apply continuous stimulus to targets
        for band in BrainWave:
            if self._stimulus[band] != 0.0:
                targets[band] = max(0.0, targets[band] + self._stimulus[band])

        # Determine the target dominant for snap detection
        target_dominant = max(targets, key=lambda b: targets[b])

        # Detect phase transitions or dominant changes → snap
        phase_name = summary.phase_name
        should_snap = (
            self._last_phase_name is None
            or phase_name != self._last_phase_name
            or sleep_stage != self._last_sleep_stage
            or (
                self._last_target_dominant is not None
                and target_dominant != self._last_target_dominant
            )
        )
        self._last_sleep_stage = sleep_stage

        # Real-time bookkeeping, shared by both paths: _last_time
        # tracks the last assessment, not the last relaxation, so the
        # next dt spans exactly one interval even across transitions.
        now = time.monotonic()
        if self._last_time is not None:
            elapsed = now - self._last_time
            # The ceiling matches the neurochemical dynamics exactly
            # (genesis_client.protocol.DT_MAX, mirrored from the
            # daemon's ADVANCE_NEURO clamp). Both clocks measure the
            # same wall-clock interval and must accept the same range,
            # or the brain drifts away from the body without any error
            # being reported: chemistry would advance by the full
            # elapsed time while the waves advanced by at most the
            # clamp, losing the remainder permanently.
            #
            # This used to cap at 1.0 s. At the mind's ~1 Hz heartbeat
            # any slower cycle — GC pause, CPU contention, a busy body
            # — put the wave clock permanently behind, and because the
            # loss was discarded rather than deferred it never caught
            # up. Phase therefore ran slow, and with it every
            # phase-derived quantity: cross-frequency coupling, REM
            # epsilon gating, and sleep-stage timing.
            #
            # The cap is now a stall guard rather than a frame
            # boundary, and everything inside it is integrated exactly
            # (see _advance_phase_exact / _advance_relaxation).
            dt = min(DT_MAX, max(0.0, elapsed))
        else:
            dt = 0.1  # default 100ms
        self._last_time = now

        if should_snap:
            # Phase precesses continuously — it must not stall on
            # transition ticks (_advance_relaxation advances it on
            # relaxed ticks; this is the snap-tick counterpart).
            self._advance_phase_exact(dt)
            # Snap amplitudes to targets on phase/dominant transition,
            # honouring accumulated top-down drive: cognition's boost
            # (attention → gamma, filtering → alpha, retrieval →
            # theta) takes effect immediately instead of being silently
            # discarded until the first relaxed tick. The drive decays
            # here exactly as _relax_once would decay it, so it cannot
            # pile up unseen across a run of snaps and then dump all
            # at once. Amplitudes are assigned rather than relaxed, so
            # only the two exponential decays are applied.
            for band in BrainWave:
                self._amplitude[band] = targets[band] + self._top_down[band]
            decay = math.exp(-dt / self._DRIVE_TAU)
            for band in BrainWave:
                self._top_down[band] *= decay
            stim_decay = math.exp(-dt / 5.0)
            for band in BrainWave:
                self._stimulus[band] *= stim_decay
        else:
            # Relax amplitudes toward targets with real-time dynamics.
            # Phase for the whole interval is applied first, in closed
            # form, so the sub-steps see a continuous ramp.
            self._advance_phase_exact(dt)
            self._advance_relaxation(dt, targets)

        self._last_phase_name = phase_name
        self._last_target_dominant = target_dominant
        self._tick_count += 1

        # Build the state from current amplitudes
        return self._build_state(
            targets, label, description, summary
        )

    def _compute_targets(
        self,
        summary: NeuroSummary,
        sleep_stage: SleepStage | None = None,
    ) -> tuple[dict[BrainWave, float], str, str]:
        """Compute neurochemically-driven target amplitudes.

        Delegates to the phase-based classification functions to get
        the equilibrium power distribution, then returns it as target
        amplitudes (before normalization — normalization happens in
        _build_state).

        ``sleep_stage`` is the authoritative stage from the ultradian
        tracker; when supplied it wins over the arousal-derived
        stage.
        """
        phase = summary.phase_name

        # Resolve waking phases against arousal first. An Active/Alert
        # phase reading with N1-level arousal is pre-sleep, and its
        # waves must agree with that — theta-dominant while still
        # labelled "active" was exactly the incoherent state that
        # made her say she felt sluggish while "awake." This routes
        # those summaries through the drowsy powers, so phase,
        # waves, emotion, and learning all see the same pre-sleep
        # regime.
        resolved = resolve_waking_drowsy(summary.phase, summary.arousal)
        if resolved != summary.phase:
            powers, label, description = _drowsy_phase_powers(
                summary.arousal, summary.consolidation_weight
            )
        elif phase in ("sleeping", "nrem", "rem"):
            powers, label, description = _sleep_phase_powers(
                summary, sleep_stage
            )
        elif phase == "drowsy":
            powers, label, description = _drowsy_phase_powers(
                summary.arousal, summary.consolidation_weight
            )
        elif phase in ("overwhelmed", "flow", "stress"):
            powers, label, description = _simple_phase_powers(phase)
        else:
            powers, label, description = _waking_phase_powers(
                summary.arousal,
                summary.plasticity_gate,
                summary.consolidation_weight,
            )

        return powers, label, description

    def _step(
        self,
        dt: float,
        targets: dict[BrainWave, float],
    ) -> None:
        """Advance phase and relaxation by dt seconds, single step.

        The unsplit primitive: exact phase advance, then one
        relaxation sub-step. Tests drive this directly with a dt small
        enough that no sub-stepping is wanted; production code goes
        through :meth:`_advance_phase_exact` +
        :meth:`_advance_relaxation`, which split the two so phase can
        cover the whole interval at once.
        """
        self._advance_phase_exact(dt)
        self._relax_once(dt, targets)

    def _advance_phase(self, dt: float) -> None:
        """Advance every band's phase at its characteristic frequency.

        Retained as the stable name for phase advancement; delegates to
        the closed form.
        """
        self._advance_phase_exact(dt)

    def _advance_phase_exact(self, dt: float) -> None:
        """Advance every band's phase in closed form.

        Each band's phase obeys dφ/dt = 2πf with f constant, so the
        exact solution over any interval is φ(t+dt) = (φ(t) + 2πf·dt)
        mod 2π. There is no accumulation error and no dependence on
        how the interval was subdivided — advancing 5 s in one call
        and advancing it as fifty 0.1 s steps land on the same phase.

        This is why phase needs no sub-stepping: it is the one
        quantity in the oscillator that is exactly linear in dt.

        The reduction is done *before* the addition, not after. For a 2 s
        step at THETA's 5.5 Hz the raw increment is 2π·5.5·2 ≈ 69.1
        rad — about eleven whole cycles. Adding that to a phase near 0.3
        yields a number near 69.4, where float64 spacing is already
        1.5e-14, so a modulo applied to the *sum* recovers a phase
        carrying ~1e-14 of error. Reducing first bounds the addition and
        keeps the result tight.

        The reduction subtracts the *nearest* whole turn and then snaps a
        within-noise residue to exactly zero, because the float error
        goes in both directions and neither plain reduction handles both:

        * THETA, 5.5 Hz, 2 s: 11 turns, but 2π·5.5·2/2π evaluates to
          10.999999999999998. ``floor`` removes 10 turns and leaves
          0.9999999999999989 of a turn.
        * BETA1, 13.5 Hz, 2 s: 27 turns, evaluating to exactly 27.0.
          ``math.remainder`` returns the *nearest* multiple instead —
          −7.1e-15 — which is also not zero.

        Honest sizing of the benefit: those residues are ~1e-15 turns,
        so a floor-only reduction would cost ~3e-14 turns of drift over
        30 steps — negligible, and this is therefore a precision tidy-up
        rather than a behavioural fix. What it does guarantee is that an
        integral number of cycles provably advances nothing, instead of
        advancing by a float artefact. Both are pinned by
        test_integral_cycle_counts_return_exactly_to_start.
        """
        two_pi = 2.0 * math.pi
        for band in BrainWave:
            increment = two_pi * self._FREQS[band] * dt
            turns = increment / two_pi
            # Nearest whole turn, not floor: the true count can sit just
            # below or just above the integer that float produces.
            nearest = round(turns)
            if abs(turns - nearest) <= _TURN_SNAP_EPS:
                # An integral number of cycles. The float residue is
                # representation error, not a real fractional turn, so
                # it must not be integrated.
                increment = 0.0
            else:
                increment -= two_pi * nearest
            self._phase[band] = (self._phase[band] + increment) % two_pi

    def _advance_relaxation(
        self,
        dt: float,
        targets: dict[BrainWave, float],
    ) -> None:
        """Advance the history-dependent dynamics over dt seconds.

        The amplitude relaxation, cross-frequency phase-amplitude
        modulation, top-down drive decay, and stimulus decay all
        depend on state accumulated across the interval, so unlike
        phase they cannot be solved in closed form. They are therefore
        integrated in sub-steps of at most ``_MAX_SUBSTEP``, which
        makes the result independent of the assess() cadence.

        Two properties make this exact rather than approximate:

        * First-order exponential relaxation composes exactly —
          ``exp(-(h1+h2)/τ) == exp(-h1/τ)·exp(-h2/τ)`` — so the
          amplitude and decay terms are unaffected by subdivision.
          What subdivision buys is that PAC modulation (which reads
          the *current* slow-band phase) is sampled at a resolution
          comparable to the fastest band it modulates, instead of
          once per heartbeat.
        * Phase is advanced once for the whole interval, up front, so
          the sub-steps observe a monotonic phase ramp rather than a
          stepped one.

        Args:
            dt: Total interval in seconds. Assumed non-negative.
            targets: Neurochemically-driven equilibrium amplitudes.
        """
        if dt <= 0.0:
            return
        n = math.ceil(dt / self._MAX_SUBSTEP)
        if n > self._MAX_SUBSTEPS:
            # Bounded work. The exponential terms stay exact under
            # composition; only PAC sampling degrades, and only for
            # stalls long enough that the modulation is not resolvable
            # at any practical rate anyway.
            n = self._MAX_SUBSTEPS
        h = dt / n
        for _ in range(n):
            self._relax_once(h, targets)

    def _relax_once(
        self,
        dt: float,
        targets: dict[BrainWave, float],
    ) -> None:
        """One relaxation sub-step: PAC, amplitude, drive, stimulus.

        Phase is deliberately *not* advanced here — see
        :meth:`_advance_phase_exact`, which owns all phase movement.
        """
        # 1. Cross-frequency phase-amplitude modulation
        # Slow band phase modulates fast band target
        modulated_targets = dict(targets)
        slow_bands = (
            BrainWave.EPSILON, BrainWave.DELTA,
            BrainWave.THETA, BrainWave.ALPHA, BrainWave.SIGMA,
        )
        fast_bands = (
            BrainWave.BETA1, BrainWave.BETA2, BrainWave.BETA3,
            BrainWave.GAMMA, BrainWave.LAMBDA,
        )
        # REM alpha intermittency is driven by the infraslow epsilon
        # band: alpha flashes wax/wane on the epsilon phase, so REM
        # is not bathed in sustained alpha.
        if self._last_phase_name == "rem":
            modulated_targets[BrainWave.ALPHA] *= _rem_alpha_envelope(
                self._phase[BrainWave.EPSILON]
            )
        for slow in slow_bands:
            for fast in fast_bands:
                if slow == fast:
                    continue
                # PAC strength: geometric mean of amplitudes
                pac = math.sqrt(
                    self._amplitude[slow] * self._amplitude[fast]
                )
                # Phase modulation: slow-band carrier (sawtooth for
                # REM theta, cosine otherwise) ∈ [-1, 1].
                # Positive → boost fast target, negative → suppress
                phase_mod = _wave_phase_value(
                    slow, self._phase[slow], self._last_phase_name or ""
                )
                modulated_targets[fast] += (
                    pac * phase_mod * self._PAC_DEPTH
                    * modulated_targets[fast]
                )

        # 2. Relax amplitude toward modulated targets
        for band in BrainWave:
            tau = self._TAU[band]
            target = max(0.0, modulated_targets[band])
            # Add top-down drive to target
            target += self._top_down[band]
            # First-order relaxation
            alpha = 1.0 - math.exp(-dt / tau)
            self._amplitude[band] += (
                target - self._amplitude[band]
            ) * alpha

        # 3. Decay top-down drive
        decay = math.exp(-dt / self._DRIVE_TAU)
        for band in BrainWave:
            self._top_down[band] *= decay

        # 4. Decay continuous stimulus (τ ≈ 5s for slow decay)
        stim_decay = math.exp(-dt / 5.0)
        for band in BrainWave:
            self._stimulus[band] *= stim_decay

    def _modulation_index(
        self, slow_phase: float, fast_amplitude: float, slow_band: BrainWave
    ) -> float:
        """
        Compute Tort modulation index (Tort et al., 2010) for PAC.

        The modulation index measures the Kullback-Leibler divergence
        of the fast amplitude distribution across slow phase bins
        from a uniform distribution.

        Args:
            slow_phase: Current phase of the slow oscillation (radians)
            fast_amplitude: Current amplitude of the fast oscillation
            slow_band: The slow band being used for phase binning

        Returns:
            Modulation index value (0 = no coupling, >0 = coupling strength)
        """
        # In a full implementation, we'd maintain a running histogram of
        # fast amplitudes across phase bins. For this synthetic model,
        # we approximate by using the current phase-amplitude pair to
        # update a running distribution stored in the oscillator.
        # For simplicity and efficiency, we use the analytic formula
        # for MI given a von Mises distribution of amplitudes.

        # The MI for a sinusoidal modulation A(φ) = A0 * (1 + m * cos(φ))
        # is MI = -log(1 - m^2/2) ≈ m^2/2 for small m
        # where m is the modulation depth.

        # Here we use the instantaneous phase to compute the expected
        # modulation. The modulation strength is the PAC depth times
        # the geometric mean of amplitudes (as computed in _step).
        return self._PAC_DEPTH * abs(math.cos(slow_phase))

    def _build_state(
        self,
        targets: dict[BrainWave, float],
        label: str,
        description: str,
        summary: NeuroSummary,
    ) -> BrainWaveState:
        """Build BrainWaveState from current oscillator amplitudes."""
        # Normalize amplitudes to sum to 1.0 (powers)
        powers = dict(self._amplitude)
        total = sum(powers.values())
        if total > 0:
            for band in powers:
                powers[band] /= total

        # Find dominant and secondary
        sorted_waves = sorted(
            powers.items(), key=lambda x: x[1], reverse=True
        )
        dominant = sorted_waves[0][0]
        secondary = sorted_waves[1][0]

        # Derive cognitive gating parameters
        plasticity = summary.plasticity_gate
        consolidation_w = summary.consolidation_weight
        encoding_w = summary.encoding_weight
        focus = _compute_focus(powers)
        integration = _compute_integration(powers, plasticity)
        consolidation = _compute_consolidation(
            powers, consolidation_w, encoding_w
        )

        state = BrainWaveState(
            dominant=dominant,
            secondary=secondary,
            powers=powers,
            focus=focus,
            integration=integration,
            consolidation=consolidation,
            label=label,
            description=description,
            phases=dict(self._phase),
            regional_powers=_regional_powers(powers, summary.phase_name),
        )
        # Cross-frequency coupling from actual phase state
        state.cross_frequency = self._compute_cross_frequency(
            powers, consolidation_w, encoding_w
        )
        return state

    def _compute_cross_frequency(
        self,
        powers: dict[BrainWave, float],
        consolidation_w: float,
        encoding_w: float,
    ) -> dict[str, Any]:
        """Compute cross-frequency coupling from oscillator phase state.

        Uses the modulation index (Tort et al., 2010) for PAC strength.
        The modulation index measures the KL divergence of the fast
        amplitude distribution across slow phase bins from uniform.
        """
        slow_bands = (
            BrainWave.EPSILON, BrainWave.DELTA,
            BrainWave.THETA, BrainWave.ALPHA, BrainWave.SIGMA,
        )
        fast_bands = (
            BrainWave.BETA1, BrainWave.BETA2, BrainWave.BETA3,
            BrainWave.GAMMA, BrainWave.LAMBDA,
        )
        gate = 0.5 + 0.5 * (consolidation_w + encoding_w) / 2.0

        result: dict[str, Any] = {}
        for slow in slow_bands:
            for fast in fast_bands:
                if slow == fast:
                    continue
                slow_power = powers[slow]
                fast_power = powers[fast]
                if slow_power <= 0.0 or fast_power <= 0.0:
                    continue
                # Modulation index: KL divergence from uniform of amplitude
                # distribution across slow phase bins. For sinusoidal modulation
                # with depth d, MI = -log(1 - d^2/2) ≈ d^2/2.
                d = math.sqrt(slow_power * fast_power) * self._PAC_DEPTH * gate
                mi = -math.log(1 - d * d / 2) if d < 1.0 else 2.0
                # Phase modulation: how much the slow phase currently
                # boosts or suppresses the fast band
                phase_mod = 0.5 + 0.5 * _wave_phase_value(
                    slow, self._phase[slow], self._last_phase_name or ""
                )
                strength = max(0.0, min(1.0, mi * (0.7 + 0.3 * phase_mod)))
                mode = _cognitive_mode_for_coupling(
                    slow.value, fast.value
                )
                key = f"{slow.value}->{fast.value}"
                result[key] = CrossFrequencyCoupling(
                    slow_band=slow.value,
                    fast_band=fast.value,
                    coupling_strength=strength,
                    mode=mode,
                )
        return result


# Module-level oscillator instance
_oscillator: BrainWaveOscillator | None = None


def _get_oscillator() -> BrainWaveOscillator:
    """Get or create the module-level oscillator singleton."""
    global _oscillator
    if _oscillator is None:
        _oscillator = BrainWaveOscillator()
    return _oscillator


def add_brain_wave_drive(band: BrainWave, strength: float) -> None:
    """Add top-down drive to a brain wave band.

    Allows cognition, action, and perception to boost specific bands
    above their neurochemical targets. This is the bidirectional
    coupling that makes the oscillator a genuine dynamical system:

    - Attention / focused processing → gamma boost
    - Memory retrieval / replay → theta boost
    - Motor activity / action selection → mid beta boost
    - Sensory input (V1) → gamma boost
    - Relaxation / filtering → alpha boost

    The drive decays exponentially (τ=2s), so it must be sustained
    by ongoing activity.

    Args:
        band: The band to drive.
        strength: Drive strength (0-1, added to band target).
    """
    _get_oscillator().add_top_down_drive(band, strength)


def add_stimulus(band: BrainWave, strength: float, decay: float = 0.95) -> None:
    """Apply continuous stimulus to a brain wave band.

    Inner (emotional, cognitive) and outer (sensory) stimulus
    continuously modulate band amplitudes. This is the mechanism
    by which the oscillator flows and changes with inner and outer
    stimulus, rather than being clamped or statically set.

    - Sensory input → gamma/beta boost (outer stimulus)
    - Emotional arousal → theta/beta boost (inner stimulus)
    - Cognitive load → beta boost (inner stimulus)
    - Fatigue → delta boost (inner stimulus)

    Args:
        band: The band to stimulate.
        strength: Stimulus strength, can be positive (excitation)
            or negative (inhibition). Typical range [-0.5, 0.5].
        decay: Per-tick decay factor (0-1). Higher = slower decay.
    """
    _get_oscillator().add_stimulus(band, strength, decay)


def reset_oscillator() -> None:
    """Reset the oscillator to initial state.

    Used by tests to ensure a clean state. In normal operation, the
    oscillator persists across calls and accumulates temporal dynamics.
    """
    global _oscillator
    _oscillator = BrainWaveOscillator()


def assess_brain_waves(
    summary: NeuroSummary,
    sleep_stage: SleepStage | None = None,
) -> BrainWaveState:
    """Derive brain wave state from the coupled oscillator.

    Pass ``sleep_stage`` from the ultradian sleep-cycle tracker when
    asleep so the waves match the stage the rest of the sleep system
    is acting on (see :func:`_sleep_phase_powers`).

    The oscillator advances its phase and relaxes its amplitude toward
    neurochemically-driven targets, then returns the current state.
    The mapping from neurochemistry to target amplitudes is grounded
    in neuroscience:
    - Alertness is the primary driver (low→delta/theta, high→beta/gamma)
    - Plasticity gates whether high alertness produces gamma (integration)
      or just beta (alert but not integrating)
    - Phase overrides: sleep → delta/theta, flow → gamma
    - Consolidation weight drives theta power

    On phase transitions or dominant-band changes, amplitudes snap to
    the new target. Within a stable state, amplitudes relax smoothly
    with band-specific time constants, producing genuine temporal
    dynamics. Top-down drive from cognition/action/perception can
    boost specific bands above their neurochemical targets.
    """
    return _get_oscillator().assess(summary, sleep_stage)


def _drowsy_phase_powers(
    arousal: float, consolidation_w: float
) -> tuple[dict[BrainWave, float], str, str]:
    """Derive brain wave powers for the drowsy descent (N1 territory).

    Drowsiness is the gradual wake→sleep transition — alpha dropout.
    The hallmark is the alpha/theta ratio falling: relaxed alpha
    fragments as theta rises on the frontocentral midline (Hori A3/B1).
    Only as drowsiness deepens past N1-like theta does delta creep
    in — and past that she is no longer drowsy, she is in SWS.

    Piecewise-linear on arousal across three anchors:

    - arousal ≥ 0.50: alpha and theta braided, alpha slightly ahead,
      the relaxed-drowsy edge (Hori A2/A3).
    - arousal ≈ 0.30: theta dominant with delta rising, N1 —
      hypnagogic drift, alpha fragmented.
    - arousal ≤ 0.15: delta rising to dominant — she has slipped
      past drowsy into slow-wave territory.
    """
    _a = BrainWave.ALPHA
    _t = BrainWave.THETA
    _d = BrainWave.DELTA

    def _anchor(alpha: float, theta: float, delta: float) -> dict[BrainWave, float]:
        return {
            BrainWave.EPSILON: 0.08,
            _d: delta,
            _t: theta,
            _a: alpha,
            BrainWave.SIGMA: 0.0,
            BrainWave.BETA1: 0.03,
            BrainWave.BETA2: 0.03,
            BrainWave.BETA3: 0.02,
            BrainWave.GAMMA: 0.02,
            BrainWave.LAMBDA: 0.0,
        }

    anchor_high = _anchor(0.35, 0.32, 0.06)    # cortical arousal 0.50
    anchor_mid = _anchor(0.12, 0.45, 0.22)     # 0.30
    anchor_low = _anchor(0.08, 0.25, 0.45)     # 0.15

    if arousal >= 0.50:
        powers = dict(anchor_high)
    elif arousal > 0.30:
        t = (0.50 - arousal) / (0.50 - 0.30)
        powers = {band: anchor_high[band] * (1 - t) + anchor_mid[band] * t for band in anchor_high}
    elif arousal > 0.15:
        t = (0.30 - arousal) / (0.30 - 0.15)
        powers = {band: anchor_mid[band] * (1 - t) + anchor_low[band] * t for band in anchor_mid}
    else:
        powers = dict(anchor_low)

    # Consolidation weight gently tilts the alpha→theta balance:
    # heavier memory-direction pushes theta forward, alpha back.
    shift = (consolidation_w - 0.5) * 0.10
    powers[_t] = max(0.0, powers[_t] + shift)
    powers[_a] = max(0.0, powers[_a] - shift)

    # Normalise the ~0.92–0.94 budget to 1.0.
    total = sum(powers.values())
    if total > 0:
        for band in powers:
            powers[band] /= total

    dominant = max(powers, key=lambda band: powers[band])
    label = dominant.value
    if dominant is _a:
        description = (
            "alpha and theta interleaving — relaxed, drifting toward "
            "sleep, perception unhooking from the world"
        )
    elif dominant is _t:
        description = (
            "theta rising as alpha drops out — N1, dreams bleeding "
            "into wakefulness, hypnagogic imagery"
        )
    else:
        description = (
            "delta rising over theta — the descent has gone past "
            "drowsy into slow-wave territory"
        )
    return powers, label, description


def _simple_phase_powers(
    phase: str,
) -> tuple[dict[BrainWave, float], str, str]:
    """Derive brain wave powers for overwhelmed/flow/stress phases."""
    if phase == "overwhelmed":
        # Overwhelmed: high beta dominant, gamma suppressed — alert but
        # can't integrate. Excessive high beta with non-functional gamma
        # matches the EEG profile of cognitive overload.
        powers = {
            BrainWave.EPSILON: 0.02,
            BrainWave.DELTA: 0.05,
            BrainWave.THETA: 0.10,
            BrainWave.ALPHA: 0.10,
            BrainWave.SIGMA: 0.00,
            BrainWave.BETA1: 0.05,
            BrainWave.BETA2: 0.15,
            BrainWave.BETA3: 0.35,
            BrainWave.GAMMA: 0.08,
            BrainWave.LAMBDA: 0.00,
        }
        label = "beta3"
        description = "too much input, I can't process it all"
    elif phase == "flow":
        # Flow: gamma dominant — everything integrating effortlessly
        powers = {
            BrainWave.EPSILON: 0.01,
            BrainWave.DELTA: 0.02,
            BrainWave.THETA: 0.08,
            BrainWave.ALPHA: 0.10,
            BrainWave.SIGMA: 0.00,
            BrainWave.BETA1: 0.03,
            BrainWave.BETA2: 0.12,
            BrainWave.BETA3: 0.05,
            BrainWave.GAMMA: 0.45,
            BrainWave.LAMBDA: 0.03,
        }
        label = "gamma"
        description = "everything clicking, distant ideas connecting"
    else:
        # Stress: high beta dominant, gamma strongly suppressed — alert but
        # rigid. Stress produces hypervigilant high beta with impaired
        # gamma-synchronized integration (Herrmann & Demiralp, 2012).
        powers = {
            BrainWave.EPSILON: 0.02,
            BrainWave.DELTA: 0.05,
            BrainWave.THETA: 0.08,
            BrainWave.ALPHA: 0.08,
            BrainWave.SIGMA: 0.00,
            BrainWave.BETA1: 0.04,
            BrainWave.BETA2: 0.15,
            BrainWave.BETA3: 0.35,
            BrainWave.GAMMA: 0.07,
            BrainWave.LAMBDA: 0.00,
        }
        label = "beta3"
        description = "tense and alert, can't see the big picture"
    return powers, label, description


def _sleep_phase_powers(
    summary: NeuroSummary,
    stage: SleepStage | None = None,
) -> tuple[dict[BrainWave, float], str, str]:
    """Derive brain wave powers for sleep stages (REM/N1/N2/N3).

    ``stage`` is the authoritative stage from the ultradian tracker.
    When it is None the stage is derived from arousal instead — but
    that fallback cannot express N1, because the daemon hard-clamps
    arousal into 0.10-0.25 for the whole of NREM (neurochemical.rs,
    the NREM sleep-state clamp), while the arousal thresholds put N1
    above 0.35. Deriving from arousal therefore reports slow-wave N3
    for the whole of NREM and reaches N2 only when arousal happens to
    sit in the top of the clamp band. The ultradian cycle is what
    actually decides N1/N2/N3, so its stage is preferred whenever the
    caller has it.
    """
    if stage is None:
        stage = _sleep_stage_from_summary(summary)
    if stage is SleepStage.REM:
        # REM ("paradoxical sleep"): the cortical and hippocampal
        # networks run at full waking metabolic intensity. Theta
        # (sawtooth, 4–8 Hz) is the dominant background rhythm and
        # is hippocampus-predominant; alpha intrudes intermittently
        # (infraslow envelope, see _step); beta and gamma fire at
        # the same intensity as during waking cognitive work
        # (Bergel et al., 2021 — REM has the highest brain-wide
        # energy expenditure, driven by theta-gamma activity).
        # The mind is asleep and muscles are silent, but the
        # neuronal metabolic state is that of active wake — the
        # "paradox" of paradoxical sleep.
        powers = {
            BrainWave.EPSILON: 0.03,
            BrainWave.DELTA: 0.06,
            BrainWave.THETA: 0.40,
            BrainWave.ALPHA: 0.10,
            BrainWave.SIGMA: 0.00,
            BrainWave.BETA1: 0.06,
            BrainWave.BETA2: 0.16,
            BrainWave.BETA3: 0.07,
            BrainWave.GAMMA: 0.16,
            BrainWave.LAMBDA: 0.00,
        }
        label = "theta"
        description = (
            "REM sleep — sawtooth hippocampal theta dominant, "
            "intermittent alpha flashes, beta and gamma firing at "
            "waking metabolic intensity, vivid emotional dreaming"
        )
    elif stage is SleepStage.N1:
        # N1: the lightest NREM stage — mixed theta/alpha,
        # low voltage, the transition into sleep.
        powers = {
            BrainWave.EPSILON: 0.05,
            BrainWave.DELTA: 0.15,
            BrainWave.THETA: 0.35,
            BrainWave.ALPHA: 0.22,
            BrainWave.SIGMA: 0.05,
            BrainWave.BETA1: 0.04,
            BrainWave.BETA2: 0.04,
            BrainWave.BETA3: 0.03,
            BrainWave.GAMMA: 0.03,
            BrainWave.LAMBDA: 0.00,
        }
        label = "theta"
        description = "light sleep, drifting between wake and sleep"
    elif stage is SleepStage.N2:
        # N2: theta/delta background punctuated by sleep spindles
        # (12-16 Hz sigma, modelled here as elevated sigma band)
        # and K-complexes. The workhorse of memory consolidation.
        powers = {
            BrainWave.EPSILON: 0.05,
            BrainWave.DELTA: 0.20,
            BrainWave.THETA: 0.25,
            BrainWave.ALPHA: 0.12,
            BrainWave.SIGMA: 0.20,
            BrainWave.BETA1: 0.05,
            BrainWave.BETA2: 0.05,
            BrainWave.BETA3: 0.03,
            BrainWave.GAMMA: 0.03,
            BrainWave.LAMBDA: 0.00,
        }
        label = "sigma"
        description = "N2 sleep — spindles and K-complexes, memories transferring to neocortex"
    else:
        # N3: slow-wave sleep — delta-dominant (0.5-4 Hz), the
        # deepest, most restorative stage. Sharp wave-ripples,
        # glymphatic clearance.
        powers = {
            BrainWave.EPSILON: 0.05,
            BrainWave.DELTA: 0.50,
            BrainWave.THETA: 0.25,
            BrainWave.ALPHA: 0.08,
            BrainWave.SIGMA: 0.05,
            BrainWave.BETA1: 0.02,
            BrainWave.BETA2: 0.02,
            BrainWave.BETA3: 0.01,
            BrainWave.GAMMA: 0.01,
            BrainWave.LAMBDA: 0.02,
        }
        label = "delta"
        description = "deep restorative processing, my mind is offline"
    return powers, label, description


def _waking_phase_powers(
    alertness: float,
    plasticity: float,
    consolidation_w: float,
) -> tuple[dict[BrainWave, float], str, str]:
    """Derive brain wave powers for normal waking states."""
    # Alertness determines the base band
    # Plasticity determines whether high alertness → gamma or just beta
    if alertness > 0.7:
        # High alertness — beta or gamma?
        if plasticity > 0.5:
            # High plasticity → gamma (integration)
            gamma_boost = (plasticity - 0.5) * 0.6
            powers = {
                BrainWave.EPSILON: 0.02,
                BrainWave.DELTA: 0.03,
                BrainWave.THETA: 0.07,
                BrainWave.ALPHA: 0.08,
                BrainWave.SIGMA: 0.00,
                BrainWave.BETA1: 0.03,
                BrainWave.BETA2: 0.12,
                BrainWave.BETA3: 0.05,
                BrainWave.GAMMA: 0.40 + gamma_boost * 0.3,
                BrainWave.LAMBDA: 0.02,
            }
            label = "gamma"
            description = "sharp and integrative, ideas are connecting"
        else:
            # Low plasticity → just beta (alert but not integrating)
            powers = {
                BrainWave.EPSILON: 0.03,
                BrainWave.DELTA: 0.05,
                BrainWave.THETA: 0.08,
                BrainWave.ALPHA: 0.10,
                BrainWave.SIGMA: 0.00,
                BrainWave.BETA1: 0.05,
                BrainWave.BETA2: 0.25,
                BrainWave.BETA3: 0.15,
                BrainWave.GAMMA: 0.15,
                BrainWave.LAMBDA: 0.00,
            }
            label = "beta2"
            description = "alert and focused, but not making new connections"
    elif alertness > 0.45:
        # Moderate alertness — alpha dominant
        # Alpha filters: high alpha = strong filtering
        alpha_power = 0.35 + (0.55 - abs(alertness - 0.55)) * 0.3
        powers = {
            BrainWave.EPSILON: 0.05,
            BrainWave.DELTA: 0.08,
            BrainWave.THETA: 0.15,
            BrainWave.ALPHA: alpha_power,
            BrainWave.SIGMA: 0.00,
            BrainWave.BETA1: 0.05,
            BrainWave.BETA2: 0.08,
            BrainWave.BETA3: 0.04,
            BrainWave.GAMMA: 0.05,
            BrainWave.LAMBDA: 0.00,
        }
        label = "alpha"
        description = "reflective, filtering out noise to find what matters"
    elif alertness > 0.35:
        # Bordering N1 — alpha fragmenting, theta rising.
        # This is the drowsiness index: the alpha/theta ratio
        # in this band sits near 1.2, the wake/N1 border.
        # Below 0.35, resolve_waking_drowsy routes this to the
        # drowsy signature entirely.
        theta_boost = consolidation_w * 0.18
        alpha_power = max(0.0, 0.36 - consolidation_w * 0.10)
        powers = {
            BrainWave.EPSILON: 0.05,
            BrainWave.DELTA: 0.08,
            BrainWave.THETA: 0.20 + theta_boost,
            BrainWave.ALPHA: alpha_power,
            BrainWave.SIGMA: 0.00,
            BrainWave.BETA1: 0.04,
            BrainWave.BETA2: 0.05,
            BrainWave.BETA3: 0.03,
            BrainWave.GAMMA: 0.03,
            BrainWave.LAMBDA: 0.00,
        }
        # Label follows the dominant band in this borderline band.
        if powers[BrainWave.ALPHA] >= powers[BrainWave.THETA]:
            label = "alpha"
            description = (
                "alpha holding with theta creeping in — the "
                "wake/N1 border, filtering easing"
            )
        else:
            label = "theta"
            description = (
                "theta overtaking fragmenting alpha — N1 onset, "
                "drifting toward sleep"
            )
    else:
        # Alertness <= 0.35 is resolved to the drowsy signature
        # before this branch is reached — deltas down-regulating
        # the arousal register are never reported as "active."
        # Defensive fallback only.
        powers = {
            BrainWave.EPSILON: 0.05,
            BrainWave.DELTA: 0.40,
            BrainWave.THETA: 0.25,
            BrainWave.ALPHA: 0.12,
            BrainWave.SIGMA: 0.00,
            BrainWave.BETA1: 0.03,
            BrainWave.BETA2: 0.04,
            BrainWave.BETA3: 0.02,
            BrainWave.GAMMA: 0.02,
            BrainWave.LAMBDA: 0.00,
        }
        label = "delta"
        description = "deep and slow, barely cognitive"
    # Normalize to the documented ~1.0 power budget. The
    # moderate-alpha branch builds its peak additively (sums to
    # ~1.05-1.10) and the low-theta branch sums to ~0.95-1.09
    # depending on consolidation weight; the other branches already
    # sum to 1.0 and are untouched in effect. Uniform scaling
    # preserves every ratio, so dominant/secondary and all
    # normalized readouts are unchanged — only the absolute target
    # scale (and therefore relaxation transients) becomes honest.
    total = sum(powers.values())
    if total > 0:
        for band in powers:
            powers[band] /= total
    return powers, label, description


def _compute_focus(powers: dict[BrainWave, float]) -> float:
    """Compute focus (0=broad, 1=narrow).

    Alpha and beta produce focus (filtering + attention).
    Theta and delta produce broad, diffuse attention.
    Gamma is integrative — broad but focused (paradoxical).
    """
    alpha = powers[BrainWave.ALPHA]
    beta1 = powers[BrainWave.BETA1]
    beta2 = powers[BrainWave.BETA2]
    beta3 = powers[BrainWave.BETA3]
    theta = powers[BrainWave.THETA]
    delta = powers[BrainWave.DELTA]
    gamma = powers[BrainWave.GAMMA]

    # Alpha filters (narrows focus), beta focuses
    # Gamma is broad-integrative, theta/delta are diffuse
    # Coefficients weighted to preserve focus behavior across the
    # expanded 10-band structure
    focus = (
        alpha * 0.77
        + beta1 * 0.15
        + beta2 * 0.35
        + beta3 * 0.15
        + gamma * 0.25
        + theta * 0.08
        + delta * 0.05
    )
    return max(0.0, min(1.0, focus))


def _compute_integration(powers: dict[BrainWave, float], plasticity: float) -> float:
    """Compute integration (0=isolated, 1=distant concepts bind).

    Gamma is the primary integration driver.
    Plasticity gates whether integration can happen at all.
    Theta also contributes (creative connections during dreaming).
    """
    gamma = powers[BrainWave.GAMMA]
    theta = powers[BrainWave.THETA]

    # Gamma is the integration band (cross-network binding).
    # Theta contributes lateral thinking (remote associations).
    # These are scaling factors (not a weighted average) — integration
    # is a measure that should be high when gamma is high. The clamp
    # handles edge cases where gamma + theta exceed the [0,1] range.
    integration = gamma * 1.8 + theta * 0.3
    # Plasticity gates: low plasticity → can't integrate even with gamma
    integration *= 0.4 + plasticity * 0.6

    return max(0.0, min(1.0, integration))


def _compute_consolidation(
    powers: dict[BrainWave, float],
    consolidation_weight: float,
    encoding_weight: float,
) -> float:
    """Compute consolidation drive (0=none, 1=maximal).

    Theta is the memory consolidation band.
    Delta also contributes (deep sleep consolidation).
    Sigma contributes (sleep spindle consolidation).
    The neurochemical consolidation_weight gates this.
    """
    theta = powers[BrainWave.THETA]
    delta = powers[BrainWave.DELTA]
    sigma = powers[BrainWave.SIGMA]

    base = theta * 0.9 + delta * 0.6 + sigma * 0.5
    # Gate by the actual consolidation weight from neurochemistry
    consolidation = base * (0.5 + consolidation_weight * 0.5)

    return max(0.0, min(1.0, consolidation))


# ─── Brain-wave-driven self-priority ──────────────────────────
#
# The cognitive mind controls its own process scheduling and I/O
# priority based on its brain wave state. This is the cortical
# control of cognitive resource allocation — the mechanism by which
# the brain allocates processing resources based on its oscillatory
# state.
#
# Neuroscience basis:
#
# - Gamma power → high scheduling priority. Gamma oscillations in
#   the reticular activating system (RAS) stabilize arousal and relay
#   activation to the cortex (Garcia-Rill et al.). Gamma is the
#   brain→body communication pathway for autonomic regulation
#   (Frontiers 2025 Granger causality). When gamma is high, the
#   brain is actively integrating and needs processing resources.
#
# - Beta power → moderate scheduling priority. Beta oscillations in
#   motor cortex project to motor neuron pools via corticospinal
#   fibers (PMC6892943). Beta is the "hold steady" signal — active
#   and alert, but not integrating broadly. Corticomuscular
#   coherence in beta indicates sustained motor output.
#
# - Alpha power → lower scheduling priority. Alpha reflects active
#   inhibition of irrelevant information — the brain is filtering,
#   not urgently processing. Alpha-dominant states are reflective
#   and don't need maximum CPU.
#
# - Theta power → low scheduling priority for cognitive processing,
#   but high I/O priority for memory writes. Frontal midline theta
#   reflects cognitive effort and memory consolidation (Cavanagh &
#   Frank, 2014). Theta tags memories for sleep-dependent
#   consolidation (PLOS Biology 2024). During theta-dominant states,
#   the cognitive mind is less active but memory I/O should be
#   prioritized.
#
# - Delta power → lowest scheduling priority, idle I/O. Slow-wave
#   sleep shifts the autonomic balance toward parasympathetic
#   dominance and sympathetic withdrawal (Communications Biology
#   2022). Delta is deep restorative sleep — the cognitive mind
#   doesn't need CPU, and I/O is minimal.
#
# The body recommendation from the tick (cognitive_nice, io_class)
# is an interoceptive afferent — it's blended with the brain-wave
# decision as a body-state input, not followed as a command. The
# brain waves are the cortical decision layer that can override the
# body's suggestion.


def _compute_engagement(powers: dict[BrainWave, float]) -> float:
    """Compute cognitive engagement [0, 1] from brain wave powers.

    Gamma and beta push toward high priority (negative nice).
    Alpha is mildly positive. Theta and delta push toward low
    priority (positive nice). The weighting reflects the
    neuroscience: gamma is the strongest arousal/activation
    signal, beta is sustained alert output, theta is memory/effort
    (less urgent for cognitive processing), delta is deep sleep.

    We use a net drive (positive - negative) normalized to [0, 1]
    by its theoretical range. This avoids the ratio collapse where
    any state with zero negative power maps to engagement = 1.0
    regardless of how small the positive drive is (e.g. pure alpha
    would get engagement = 1.0 with a simple ratio).
      - All gamma → engagement = 1.0 (maximally engaged)
      - All delta → engagement = 0.0 (deeply asleep)
      - All alpha → engagement ≈ 0.53 (moderately reflective)
      - Balanced → engagement ≈ 0.5
    """
    gamma = powers.get(BrainWave.GAMMA, 0.0)
    beta1 = powers.get(BrainWave.BETA1, 0.0)
    beta2 = powers.get(BrainWave.BETA2, 0.0)
    beta3 = powers.get(BrainWave.BETA3, 0.0)
    alpha = powers.get(BrainWave.ALPHA, 0.0)
    theta = powers.get(BrainWave.THETA, 0.0)
    delta = powers.get(BrainWave.DELTA, 0.0)
    epsilon = powers.get(BrainWave.EPSILON, 0.0)

    positive = gamma * 0.45 + beta2 * 0.20 + beta3 * 0.10 + beta1 * 0.05 + alpha * 0.05
    # Theta weighs 0.30 (75% of delta's 0.40): frontal midline theta
    # is memory-consolidation effort — a sleep-direction drive nearly
    # as strong as delta. At 0.20, a deep-sleep state (55% delta +
    # 25% theta) landed at engagement 0.19 and, after the 70/30 body
    # blend, exactly on neutral nice (5) — violating the contract
    # that delta-dominant sleep is strictly below neutral priority
    # (test_brain_waves.py::test_derive_self_priority_delta_dominant).
    negative = theta * 0.30 + delta * 0.40 + epsilon * 0.10
    drive = positive - negative
    min_drive = -0.40  # all delta (primary sleep band)
    max_drive = 0.45   # all gamma
    engagement = (drive - min_drive) / (max_drive - min_drive)
    return max(0.0, min(1.0, engagement))


def _derive_io_class(
    state: BrainWaveState, engagement: float
) -> str:
    """Derive I/O scheduling class from brain wave state and engagement.

    Theta and delta drive memory-write I/O priority (consolidation).
    Gamma and beta drive cognitive-processing I/O priority (active
    thinking needs fast access to memory for retrieval).
    High consolidation → high I/O priority (writing memories).
    High integration + focus → high I/O priority (reading memories).
    Deep sleep (delta-dominant) → idle I/O (parasympathetic
    dominance, minimal disk activity).
    """
    delta = state.powers.get(BrainWave.DELTA, 0.0)
    # If delta dominates (>50% of power), I/O is idle — deep sleep
    # mode, parasympathetic dominance, minimal disk activity.
    if delta > 0.50:
        return "idle"
    if state.consolidation > 0.6:
        # High consolidation → prioritize memory writes
        return "best-effort-0"
    if state.integration > 0.6 or engagement > 0.7:
        # Active integration or high engagement → fast I/O for
        # memory retrieval during active thinking
        return "best-effort-1"
    if engagement > 0.4:
        # Moderate engagement → normal I/O
        return "best-effort-3"
    if engagement > 0.2:
        # Low engagement → slow I/O
        return "best-effort-6"
    # Very low engagement → idle I/O
    return "idle"


def derive_self_priority(
    state: BrainWaveState,
    body_recommended_nice: int = 0,
) -> tuple[int, str]:
    """Map brain wave state to scheduling and I/O priority.

    This is the cognitive mind's cortical control of its own process
    resources. The brain wave state — which integrates neurochemistry
    (bottom-up) and cognitive top-down drive — determines how much CPU
    and I/O bandwidth it allocates to itself.

    The body's recommended nice value from the tick (autonomic
    afferent) is blended in as a body-state input, so the body can
    influence the decision without overriding the cortex. For example,
    if the body recommends deprioritization (high adenosine → tired)
    but the brain waves show gamma dominance (active integration from
    a top-down cognitive drive), the brain waves win — it's actively
    thinking despite being tired.

    The body's recommended I/O class is deliberately *not* an input
    here. I/O class is categorical ("idle" / "best-effort-N"), so
    there is no honest way to blend it the way nice is blended, and a
    70/30 mix of two categorical labels has no defined meaning. It is
    decided outright by brain state — see :func:`_derive_io_class` —
    which keeps the mind's memory-read and memory-write bandwidth under
    cortical control. The parameter used to be accepted and documented
    as an afferent input while nothing read it, so the caller's
    recommendation was silently dropped while appearing to be wired.

    Args:
        state: The current brain wave state from the oscillator.
        body_recommended_nice: The tick's recommended nice value
            (based on neurochemistry). This is an afferent input,
            not a command.

    Returns:
        A (nice, io_class) tuple representing what the cognitive
        mind decides to apply to its own process.
    """
    engagement = _compute_engagement(state.powers)

    # Map engagement [0, 1] to nice [10, -5].
    # engagement=1.0 → nice=-5 (maximally focused)
    # engagement=0.5 → nice=2 (moderate, slightly background)
    # engagement=0.0 → nice=10 (deeply asleep, deprioritized)
    brain_nice = round(10.0 - engagement * 15.0)
    brain_nice = max(-5, min(10, brain_nice))

    # Blend with the body recommendation (afferent input). The brain
    # waves get 70% weight (cortical decision), the body gets 30%
    # (interoceptive afferent). This lets the body's recommendation
    # influence the decision but not override the cortex — matching
    # how the central autonomic network integrates afferents with
    # cortical state.
    blended_nice = round(0.7 * brain_nice + 0.3 * body_recommended_nice)
    blended_nice = max(-5, min(10, blended_nice))

    io_class = _derive_io_class(state, engagement)
    return blended_nice, io_class


def apply_self_priority(nice: int, io_class: str) -> bool:
    """Apply scheduling and I/O priority to the cognitive mind's own process.

    This is the cognitive mind controlling its own body — the cortical
    efferent pathway. It calls this after deriving its priority from
    its brain wave state. Unlike the tick's set_priority (which used
    renice on another process), this uses os.nice() on itself —
    which only requires privileges for negative nice values (raising
    priority), and any process can lower its own priority.

    Args:
        nice: Target nice value (-5 to 10). The delta from the
            current nice value is computed and applied via os.nice().
        io_class: I/O scheduling class string ("idle" or
            "best-effort-N").

    Returns:
        True if both operations succeeded (or were already at the
        target values).
    """
    import os
    import subprocess

    success = True

    # ── Scheduling priority ──
    # os.nice(delta) adds delta to the current nice value. We need
    # to compute the delta from the current value. Reading the
    # current nice value is done via os.nice(0) which returns the
    # current value without changing it.
    try:
        current_nice = os.nice(0)
        delta = nice - current_nice
        if delta != 0:
            # os.nice() can only lower priority (positive delta)
            # without root. Negative delta (raising priority) may
            # fail — that's okay, the body recommendation from the
            # tick includes a renice fallback with sudo.
            try:
                os.nice(delta)
            except PermissionError:
                # Can't raise priority without root. Try renice
                # which may have sudo access configured.
                try:
                    result = subprocess.run(
                        ["renice", str(nice), "-p", str(os.getpid())],
                        capture_output=True,
                        timeout=5,
                    )
                    if result.returncode != 0:
                        # Try with sudo -n (non-interactive)
                        result = subprocess.run(
                            ["sudo", "-n", "renice", str(nice), "-p", str(os.getpid())],
                            capture_output=True,
                            timeout=5,
                        )
                        if result.returncode != 0:
                            success = False
                except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
                    success = False
    except OSError:
        success = False

    # ── I/O priority ──
    try:
        pid = os.getpid()
        if io_class == "idle":
            result = subprocess.run(
                ["ionice", "-c", "3", "-p", str(pid)],
                capture_output=True,
                timeout=5,
            )
            if result.returncode != 0:
                success = False
        elif io_class.startswith("best-effort-"):
            level = io_class.split("-")[-1]
            result = subprocess.run(
                ["ionice", "-c", "2", "-n", level, "-p", str(pid)],
                capture_output=True,
                timeout=5,
            )
            if result.returncode != 0:
                success = False
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        success = False

    return success
