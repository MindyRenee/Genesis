"""Brain wave states — a coupled amplitude-phase oscillator.

## Modeling note

This is a **synthetic oscillator model** inspired by EEG, not an EEG
physiology model. Real EEG exhibits region-specific phase relationships
across cortical sources, absolute (not normalized) spectral power
measured in microvolts, and proper phase-amplitude coupling computed
via the Hilbert transform on band-passed signals. The oscillator here
uses normalized amplitudes and a single global phase per band —
except the gamma binding layer, which carries four regional phase
oscillators so cross-region locking is a measured quantity rather
than an assumed one. The claim that high gamma
and high phase-locking value (PLV) correspond to "cognitive access"
is a heuristic gating rule inspired by the global neuronal workspace
hypothesis, not a validated cognition marker. The citations
(Canolty & Knight, 2010) refer to the empirical phenomena that
motivated the design, not to claims that this oscillator reproduces
them.

Biological rhythms involve distributed circuits and correlate with
multiple tasks and states, not one-to-one cognitive labels. The
following associations are this model's control conventions:

- **Epsilon (100-200 Hz)** — a synthetic HFO/ripple band. The name
  follows the convention that puts "epsilon" above gamma for
  >~90-100 Hz LFP activity (Freeman, 2007; Buzsáki & Wang, 2012;
  Belluscio et al., 2012) — not the >250 Hz fast-ripple band, and
  unrelated to the clinical occipital lambda potential.
- **Gamma (30-80 Hz)** — the model's integration/binding control.
  Biological gamma is not a universal marker of insight or awareness.
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
- **Infraslow (0.1-0.5 Hz)** — infraslow oscillations (ISOs). Slow
  cortical potentials that modulate cortical excitability and
  coordinate large-scale network dynamics.

# How Genesis's brain waves work

Genesis's brain waves are produced by a bank of **driven phasor
oscillators** (``BrainWaveOscillator``) — not a classifier. Each
band is a real oscillation with persistent quadrature state, and its
reported power is *measured*: the band emits its real-axis trace and
a Goertzel meter — the same band math the body's field probe ran
over the substrate before the two swapped places — reads the wave's
band energy off the emitted signal, never off state.

- **Waveform**: each band carries a complex state ``z`` rotating at
  the band's characteristic frequency (delta ~2.25 Hz, theta ~5.5 Hz,
  alpha ~10 Hz, sigma ~14 Hz, low beta ~13.5 Hz, mid beta ~17.5 Hz,
  high beta ~25 Hz, gamma ~50 Hz, epsilon ~150 Hz). ``Re(z)`` is a
  genuine synthesized trace — emitted into the metering ring at
  512 Hz, it is the signal the band meter integrates.
- **Amplitude** is the radius |z|. Neurochemical targets set the
  radius the envelope relaxes toward, with the band's time constant
  as its first-order growth/ring-down rate — so a state change is an
  oscillation swelling or dying away.
- **Phase** is *measured* from the state (``arg z``), not advanced by
  a stopwatch. Under the stochastic drive it diffuses and wanders the
  way real rhythm phase does — a kick (``perturb_phase``) is a real
  rotation of the oscillator's position in its cycle.
- **Stochastic drive**: complex Gaussian noise keeps every band a
  driven quasicycle rather than a pinned setpoint — envelopes wander
  and phases diffuse, the amplitude texture stochastic neural
  oscillators actually show (Pérez-Cervera et al., 2021:
  Rayleigh-like amplitude density around the cycle).
- **Cross-frequency coupling** emerges from phase-amplitude
  interaction: the slow band's measured phase modulates the fast
  band's drive — hence its real oscillatory power (phase-amplitude
  coupling, the canonical mechanism for cross-scale neural
  coordination; Canolty & Knight, 2010).
- **Top-down drive** from cognition, action, and perception enlarges
  specific envelopes above their neurochemical radii. Attention →
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
  infraslow envelope, echoing quiet wakefulness rather
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
import random
import threading
import time
from collections import deque
from collections.abc import MutableMapping
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
    "perturb_wave_phase",
    "reset_oscillator",
    "reset_wave_phase",
    "resolve_waking_drowsy",
    "wave_phase_of",
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
    """Model bands. ``epsilon`` is Freeman's (2007) name for the
    >~90-100 Hz LFP band above gamma; ``infraslow`` is the ISO band.
    Neither is one of the five classical EEG band names."""

    EPSILON = "epsilon"  # 100-200 Hz — high-frequency oscillations (HFOs)
    GAMMA = "gamma"  # 30-80 Hz — integration, binding
    BETA3 = "beta3"  # 20-30 Hz — hypervigilance, stress
    BETA2 = "beta2"  # 15-20 Hz — active thinking, alert
    BETA1 = "beta1"  # 12-15 Hz — calm focus, SMR
    SIGMA = "sigma"  # 12-16 Hz — sleep spindles (NREM)
    ALPHA = "alpha"  # 8-12 Hz — relaxed reflection, filtering
    THETA = "theta"  # 4-8 Hz — memory, creativity, REM
    DELTA = "delta"  # 0.5-4 Hz — deep sleep, restoration
    INFRASLOW = "infraslow"  # 0.1-0.5 Hz — infraslow oscillations (ISOs)


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
    regional_powers: dict[str, dict[BrainWave, float]] = field(default_factory=dict)

    # Regional gamma binding layer — instantaneous phases of the four
    # regional gamma oscillators, and the windowed phase-locking value
    # computed from their rolling phase history. ``gamma_plv`` is the
    # true PLV |<exp(i*dphi)>| across all six region pairs (None when
    # the state wasn't built by a live oscillator); ``gamma_plv_pairs``
    # breaks it down per pair.
    regional_gamma_phases: dict[str, float] = field(default_factory=dict)
    gamma_plv: float | None = None
    gamma_plv_pairs: dict[str, float] = field(default_factory=dict)

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
    names (the Rust side distinguishes NREM from REM).

    For the legacy "sleeping" phase, arousal > 0.50 maps to REM
    (since the old daemon didn't distinguish NREM from REM). For the
    explicit "nrem" phase, only N1/N2/N3 are produced — the daemon
    said NREM, so it's NREM regardless of arousal. In practice the
    daemon clamps NREM arousal into a narrow band, so while a sleep
    cycle is active the ultradian tracker — not this mapping — is the
    stage authority; this is the fallback when no tracker is running:

        0.35 < arousal         → N1
        0.20 < arousal ≤ 0.35  → N2
        arousal ≤ 0.20         → N3
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
# and ~3.3e-13 at the largest count reachable within DT_MAX (epsilon,
# 150 Hz over 10 s). 1e-9 turns is three to six orders of magnitude
# above all of those, while being 6.3e-9 rad -- far below anything the
# downstream phase consumers can observe. A genuinely fractional cycle
# count (a band whose frequency times the interval is not near-integral,
# e.g. alpha at 10 Hz over 0.2 s = 2.0 exactly, or 5.5 Hz over 0.2 s =
# 1.1) sits 0.1 turns or more away, so it is never snapped.
_TURN_SNAP_EPS = 1e-9


def _band_power(samples: deque[float] | list[float], fs: float, freq: float) -> float:
    """Goertzel power of ``samples`` at ``freq`` — the field probe's
    band meter, carried into the mind.

    Second-order recurrence over the emitted trace:
    ``s₀ = x + coeff·s₁ − s₂`` with ``coeff = 2·cos(2πk/n)`` at the
    exact bin nearest ``freq``; power ``= (s₁² + s₂² − coeff·s₁·s₂)
    / n²``. The wave's reported band energy is *measured* off the
    emitted trace — the same arithmetic the body applies to the
    substrate's real signals — not read from oscillator state.

    Bands above the metering rate's Nyquist bound return 0 — an
    absent channel, not a reading of zero.
    """
    n = len(samples)
    if n < 4:
        return 0.0
    k = max(1, round(n * freq / fs))
    if k >= n / 2:
        return 0.0  # above Nyquist — absent channel
    w = 2.0 * math.pi * k / n
    coeff = 2.0 * math.cos(w)
    s1 = s2 = 0.0
    for x in samples:
        s0 = x + coeff * s1 - s2
        s2 = s1
        s1 = s0
    return (s1 * s1 + s2 * s2 - coeff * s1 * s2) / (n * n)


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
        "epsilon": "fast_ripple",
        "gamma": "integration",
        "beta3": "hypervigilance",
        "beta2": "active_thinking",
        "beta1": "calm_focus",
        "sigma": "consolidation",
        "alpha": "gating",
    }.get(fast, "coupling")
    # Slow band gives the *state* in which the gating happens
    slow_role = {
        "infraslow": "infraslow_modulation",
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


def _windowed_gamma_plv(
    samples: deque[tuple[float, tuple[float, float, float, float]]],
) -> tuple[float | None, dict[str, float]]:
    """True phase-locking value over regional gamma phase history.

    For each region pair, the PLV is the mean resultant length of the
    phase difference across the sample window —

        PLV_ij = | sum_t exp(i * (phi_i[t] - phi_j[t])) | / N

    (Lachaux et al., 1999). The mean of the six pairwise values is
    returned alongside the per-pair breakdown. ``None`` when the
    window holds too few samples — an absent channel, not a zero.
    """
    # Snapshot first: the oscillator appends to the same deque on
    # its own thread — iterating the live deque races it. The
    # snapshot itself is an iteration, so a racing append can still
    # fail it; a bounded retry lands between appends.
    window: list[tuple[float, tuple[float, float, float, float]]] = []
    for _ in range(3):
        try:
            window = list(samples)
            break
        except RuntimeError:
            continue
    if len(window) < 4:
        return None, {}
    n = len(window)
    pair_idx = [
        (i, j) for i in range(len(_GAMMA_REGIONS)) for j in range(i + 1, len(_GAMMA_REGIONS))
    ]
    pairs: dict[str, float] = {}
    for i, j in pair_idx:
        re = sum(math.cos(ph[i] - ph[j]) for _, ph in window)
        im = sum(math.sin(ph[i] - ph[j]) for _, ph in window)
        pairs[f"{_GAMMA_REGIONS[i]}-{_GAMMA_REGIONS[j]}"] = math.hypot(re, im) / n
    return sum(pairs.values()) / len(pairs), pairs


def compute_gamma_synchrony(
    state: BrainWaveState,
    plasticity: float = 0.5,
) -> GammaSynchrony:
    """Compute gamma-band phase-locking across cortical regions.

    ``synchrony`` is a true phase-locking value: the mean resultant
    length |<exp(i*delta_phi)>| of pairwise phase differences between
    the four regional gamma oscillators, measured over the
    oscillator's rolling one-second phase history (Lachaux et al.,
    1999). It is a property of the dynamics, not a function applied
    to the power spectrum — gamma amplitude and plasticity reach the
    number only through the Kuramoto coupling that entrains (or
    releases) the regional phases, so stress, sleep, and low drive
    desynchronise the measure on the coupling time constant instead
    of being multiplied out of the score.

    States not built by a live ``BrainWaveOscillator`` carry no phase
    history; for those the function falls back to the legacy
    power-weighted proxy (gamma power scaled by plasticity) — the
    honest reading a phase-less state can offer, and still just a
    proxy.

    The 0.5 cognitive-access cutoff remains an operational heuristic —
    GNW ignition is a nonlinear dynamical transition, not a fixed PLV
    constant (Dehaene, 2014) — but it now thresholds a measured
    synchronisation rather than a proxy.

    Args:
        state: The current brain wave state.
        plasticity: Neurochemical plasticity gate [0,1]. Drives the
            regional coupling upstream; also used by the fallback
            proxy for phase-less states.

    Returns:
        A ``GammaSynchrony`` with per-region powers, the measured (or
        proxied) PLV, and a cognitive-access prediction.
    """
    gamma = state.powers[BrainWave.GAMMA]

    # Per-region gamma power: global gamma with a small deterministic
    # spread so regions aren't identical (fronto-parietal dominance
    # during cognitive access — Dehaene, 2014).
    region_powers: dict[str, float] = {}
    for region in _GAMMA_REGIONS:
        hub_boost = 0.1 if region in ("prefrontal", "parietal") else -0.05
        region_powers[region] = max(0.0, min(1.0, gamma + hub_boost * gamma))

    if state.gamma_plv is not None:
        # The real measurement: windowed PLV over the regional
        # oscillators' own phase history.
        plv = state.gamma_plv
    elif state.regional_gamma_phases:
        # True but instantaneous: a state carrying regional phases
        # without a history window yields a single-sample PLV.
        names = list(state.regional_gamma_phases)
        re = im = cnt = 0.0
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                d = state.regional_gamma_phases[names[i]] - state.regional_gamma_phases[names[j]]
                re += math.cos(d)
                im += math.sin(d)
                cnt += 1
        plv = math.hypot(re, im) / cnt if cnt > 0 else 0.0
    else:
        # Legacy proxy for phase-less states: gamma power scaled by
        # plasticity. Not a synchrony measurement — kept only so
        # constructed states (tests, foreign producers) still get an
        # answer.
        plv = max(0.0, min(1.0, gamma * (1.2 + plasticity * 0.8)))

    # Cognitive-access heuristic: synchrony > 0.5 predicts access.
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
# 40:8900). The epsilon band — >100 Hz LFP energy, largely
# spike-leaked (Belluscio et al., 2012) — rides strongest over
# sensory cortex, here occipital.

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
    BrainWave.INFRASLOW: {
        "prefrontal": 0.6,
        "frontal": 0.7,
        "parietal": 0.5,
        "temporal": 0.4,
        "occipital": 0.4,
        "central": 0.5,
        "hippocampus": 0.4,
    },
    BrainWave.DELTA: {
        "prefrontal": 0.8,
        "frontal": 1.0,
        "parietal": 0.6,
        "temporal": 0.5,
        "occipital": 0.5,
        "central": 0.7,
        "hippocampus": 0.4,
    },
    BrainWave.THETA: {
        "prefrontal": 0.7,
        "frontal": 0.6,
        "parietal": 0.5,
        "temporal": 0.9,
        "occipital": 0.3,
        "central": 0.5,
        "hippocampus": 1.0,
    },
    BrainWave.ALPHA: {
        "prefrontal": 0.3,
        "frontal": 0.4,
        "parietal": 0.7,
        "temporal": 0.6,
        "occipital": 1.0,
        "central": 0.4,
        "hippocampus": 0.2,
    },
    BrainWave.SIGMA: {
        "prefrontal": 0.4,
        "frontal": 0.5,
        "parietal": 0.8,
        "temporal": 0.4,
        "occipital": 0.4,
        "central": 1.0,
        "hippocampus": 0.2,
    },
    BrainWave.BETA1: {
        "prefrontal": 0.5,
        "frontal": 0.6,
        "parietal": 0.5,
        "temporal": 0.4,
        "occipital": 0.3,
        "central": 1.0,
        "hippocampus": 0.3,
    },
    BrainWave.BETA2: {
        "prefrontal": 0.8,
        "frontal": 0.9,
        "parietal": 0.6,
        "temporal": 0.5,
        "occipital": 0.4,
        "central": 0.7,
        "hippocampus": 0.3,
    },
    BrainWave.BETA3: {
        "prefrontal": 0.9,
        "frontal": 1.0,
        "parietal": 0.5,
        "temporal": 0.5,
        "occipital": 0.3,
        "central": 0.6,
        "hippocampus": 0.2,
    },
    BrainWave.GAMMA: {
        "prefrontal": 1.0,
        "frontal": 0.8,
        "parietal": 0.9,
        "temporal": 0.7,
        "occipital": 0.7,
        "central": 0.5,
        "hippocampus": 0.8,
    },
    BrainWave.EPSILON: {
        "prefrontal": 0.2,
        "frontal": 0.2,
        "parietal": 0.3,
        "temporal": 0.4,
        "occipital": 1.0,
        "central": 0.2,
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
        raw = {band: powers[band] * affinity[band][region] for band in BrainWave}
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


def _sawtooth_mean(u0: float, delta: float) -> float:
    """Exact mean of the sawtooth carrier over [u0, u0+delta].

    The carrier is affine on [0, 2π) with a single wrap, and every
    step in this module spans less than a full turn (max Δφ = ω·h ≈
    2.2 rad at the _MAX_SUBSTEPS bound), so at most one wrap can occur
    inside the interval.
    """
    two_pi = 2.0 * math.pi
    if u0 + delta <= two_pi:
        # No wrap — mean of an affine segment is its midpoint value.
        return (u0 + 0.5 * delta) / math.pi - 1.0
    # Wrap: split at 2π. Segment 1 [u0, 2π] has mean u0/(2π)
    # (midpoint of s(u0)=u0/π−1 and s(2π⁻)=1); segment 2 [0, w2]
    # has mean w2/(2π)−1.
    w1 = two_pi - u0
    w2 = delta - w1
    return (w1 * (u0 / two_pi) + w2 * (w2 / two_pi - 1.0)) / delta


def _wave_phase_mean(band: BrainWave, phase0: float, delta_phi: float, phase_name: str) -> float:
    """Exact mean of a slow band's phase carrier over a sub-step.

    The modulator driving PAC is a known periodic function of the
    slow band's phase, and that phase advances deterministically by
    Δφ = ω·h inside a step — so the *average* modulation the fast band
    actually felt over the interval is computable in closed form:

    - cosine carrier: mean(cos) = cos(φ_mid)·sinc(Δφ/2), exact for
      any Δφ — no O(h) sampling error, no dependence on how the
      interval was subdivided.
    - REM theta sawtooth: affine carrier, exact by segments.
    """
    if delta_phi <= 0.0:
        return _wave_phase_value(band, phase0, phase_name)
    if phase_name == "rem" and band is BrainWave.THETA:
        return _sawtooth_mean(phase0 % (2.0 * math.pi), delta_phi)
    half = 0.5 * delta_phi
    return math.cos(phase0 + half) * (1.0 if half < 1e-9 else math.sin(half) / half)


def _wave_phase_value(band: BrainWave, phase: float, phase_name: str) -> float:
    """The phase-carrier value for a slow band at this moment.

    REM theta uses its characteristic sawtooth carrier; every other
    band uses the symmetric cosine. This asymmetry feeds the PAC
    modulation of fast bands, so gamma during REM is timed to the
    sawtooth ramp rather than a sine peak.
    """
    if phase_name == "rem" and band is BrainWave.THETA:
        return _sawtooth(phase)
    return math.cos(phase)


def _rem_alpha_envelope(infraslow_phase: float) -> float:
    """Intermittent REM alpha — infraslow-band burst gain.

    During REM, alpha rhythms are fragmented: brief alpha intrusions
    echo quiet-wakefulness/meditative alpha, waxing and waning with
    the infraslow (~0.25 Hz) envelope (periodic ~4 s). The envelope
    is ``max(0, cos(φ))²``: a flash reaching full gain near cosine
    peak, falling to zero at cosine trough — never the sustained
    alpha of eyes-closed wake.
    """
    return max(0.0, math.cos(infraslow_phase)) ** 2


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


# ─── Coupled driven-phasor oscillator bank ──────────────────────
#
# The oscillator bank is the dynamical core. Each band is one driven
# phasor — ż = iωz + noise with the envelope relaxing toward a
# drive-prescribed radius — with persistent quadrature state (the
# oscillation is real state, not a phase counter plus a smoothed
# scalar). The Stuart–Landau normal form this bank once ran moved
# into the body: the field probe's channels now resonate with the
# substrate on that math, while the mind's reported band power is
# *measured* — Goertzel over each band's emitted trace, the meter
# the body gave up in exchange. Cross-frequency coupling emerges
# from actual phase-amplitude interaction: the slow band's measured
# phase modulates the fast band's driven radius. Top-down drive
# from cognition, action, and perception enlarges specific envelopes
# above their neurochemical radii — bidirectional coupling.


class _PhaseView(MutableMapping):
    """Live phase view over the quadrature states.

    ``view[band]`` reads the measured phase (``arg z``); writing
    ``view[band] = p`` rotates the band's state to that phase while
    keeping its amplitude — the same operation reset_phase performs,
    so state stays single-sourced in ``_z``.
    """

    __slots__ = ("_z",)

    def __init__(self, z: dict[BrainWave, complex]) -> None:
        self._z = z

    def __getitem__(self, band: BrainWave) -> float:
        z = self._z[band]
        # A collapsed oscillation has no defined phase — a band driven
        # to (near-)zero radius leaves z in denormal residue whose arg
        # is float junk, and a silenced rhythm physically *has* no
        # phase to measure. Report the canonical 0 so the readout stays
        # meaningful and chunking-independent; the state itself is
        # untouched and revives under fresh drive.
        if z.real * z.real + z.imag * z.imag < 1e-18:
            return 0.0
        return math.atan2(z.imag, z.real) % (2.0 * math.pi)

    def __setitem__(self, band: BrainWave, phase: float) -> None:
        r = abs(self._z[band])
        self._z[band] = complex(r * math.cos(phase), r * math.sin(phase))

    def __delitem__(self, band: BrainWave) -> None:
        raise TypeError("oscillator bands are fixed")

    def __iter__(self):
        return iter(self._z)

    def __len__(self) -> int:
        return len(self._z)


class _AmplitudeView(MutableMapping):
    """Live amplitude view over the quadrature states.

    ``view[band]`` reads |z|; writing rescales the band's state to
    that radius while keeping its phase (a zero state starts on the
    real axis).
    """

    __slots__ = ("_z",)

    def __init__(self, z: dict[BrainWave, complex]) -> None:
        self._z = z

    def __getitem__(self, band: BrainWave) -> float:
        return abs(self._z[band])

    def __setitem__(self, band: BrainWave, amp: float) -> None:
        z = self._z[band]
        r = abs(z)
        a = max(0.0, amp)
        self._z[band] = z * (a / r) if r > 0.0 else complex(a, 0.0)

    def __delitem__(self, band: BrainWave) -> None:
        raise TypeError("oscillator bands are fixed")

    def __iter__(self):
        return iter(self._z)

    def __len__(self) -> int:
        return len(self._z)


class BrainWaveOscillator:
    """Bank of driven phasor oscillators — one per brain wave band.

    Each band integrates a driven phasor

        ż = iω·z + σ·(ξ_x + i·ξ_y)dt^{1/2},   r → a at rate 1 − e^{−h/τ}

    over the interval between assess() calls, and emits its real-axis
    trace into a metering ring: reported band power is *measured*
    (Goertzel over the emitted trace — the band math the body ran
    before the two layers swapped), never read off state:

    - **Oscillation**: ``z = x + iy`` rotates at ω = 2πf_band. The
      wave is a real dynamical state — ``Re(z)`` is the band's
      synthesized trace — so a kick or a target change produces a
      genuine transient: overshoot, ring-down, phase slip.
    - **Amplitude** is the driven radius |z|. The neurochemical
      target is the *radius a* the envelope relaxes toward, with
      τ_band setting the first-order approach — slow bands ring long
      (delta τ=8s), fast bands re-seat quickly (gamma τ=0.5s).
    - **Phase** is measured — ``arg z`` — so it precesses at the band
      frequency and diffuses under the noise drive, instead of being
      a fixed-rate integrator. Phase perturbations rotate z itself.
    - **Stochastic drive**: σ ∝ A (+ a small spontaneous floor) makes
      the envelope wander ~η·A around its radius — the fluctuating
      band power real recordings show, rather than a clamped
      setpoint (Deco et al., 2017; Pérez-Cervera et al., 2021).
    - **Cross-frequency coupling** emerges from phase-amplitude
      interaction: the slow band's measured phase modulates the fast
      band's radius. When theta is at its peak, gamma's envelope gets
      a boost; when theta is at its trough, gamma is suppressed. This
      is the canonical PAC mechanism acting on real power.
    - **Top-down drive** from cognition/action/perception enlarges
      specific envelopes above their neurochemical radii. Attention →
      gamma, memory retrieval → theta, motor activity → mid beta.
      The drive decays exponentially (τ=2s), so it must be
      sustained by ongoing activity.

    Rotation is closed-form for any step size and the noise kick's
    variance is ∝ step, so a subdivided interval and a single step
    land on the same deterministic trajectory up to the same noise
    statistics; only the amplitude-coupled PAC sampling resolves
    finer with sub-steps. This produces genuine oscillatory state
    (frequency, phase, amplitude) from which
    measured band powers, cross-frequency coupling, and cognitive
    gating parameters are derived.
    """

    # Characteristic frequencies (Hz) — center of each band
    _FREQS: ClassVar[dict[BrainWave, float]] = {
        BrainWave.INFRASLOW: 0.25,  # 0.1-0.5 Hz
        BrainWave.DELTA: 2.25,  # 0.5-4 Hz
        BrainWave.THETA: 5.5,  # 4-8 Hz
        BrainWave.ALPHA: 10.0,  # 8-12 Hz
        BrainWave.SIGMA: 14.0,  # 12-16 Hz
        BrainWave.BETA1: 13.5,  # 12-15 Hz
        BrainWave.BETA2: 17.5,  # 15-20 Hz
        BrainWave.BETA3: 25.0,  # 20-30 Hz
        BrainWave.GAMMA: 50.0,  # 30-80 Hz
        BrainWave.EPSILON: 150.0,  # 100-200 Hz
    }

    # Radial time constants (seconds) — the envelope's ring-down /
    # spin-up scale: the first-order approach r → a advances
    # 1 − e^{−h/τ} of the remaining gap per step. Slow bands carry
    # their rhythm long (delta τ=8s); fast bands re-seat quickly
    # (gamma τ=0.5s).
    _TAU: ClassVar[dict[BrainWave, float]] = {
        BrainWave.INFRASLOW: 12.0,
        BrainWave.DELTA: 8.0,
        BrainWave.THETA: 4.0,
        BrainWave.ALPHA: 2.0,
        BrainWave.SIGMA: 1.5,
        BrainWave.BETA1: 1.2,
        BrainWave.BETA2: 1.0,
        BrainWave.BETA3: 0.8,
        BrainWave.GAMMA: 0.5,
        BrainWave.EPSILON: 0.3,
    }

    # Top-down drive decay time constant (seconds)
    _DRIVE_TAU: ClassVar[float] = 2.0

    # ── Stochastic drive ──────────────────────────────────────────
    # Fractional envelope wandering: the radius fluctuates with
    # stationary std ≈ _TEXTURE·A around the driven radius. 0.10 gives
    # the visibly bursty texture of real band power without letting
    # noise re-rank the distribution — the dominance ordering the
    # rest of the mind reads is a target-level property, not a
    # sample of the jitter.
    _TEXTURE: ClassVar[float] = 0.10
    # Spontaneous floor in the noise scale — a band driven to zero
    # still carries faint background activity (real EEG never reads
    # dead silence), so its phase diffuses slowly rather than
    # freezing at its last angle.
    _NOISE_FLOOR: ClassVar[float] = 0.02
    # Driven-radius floor: the radial approach reads a target radius,
    # floored far below anything a target would legitimately request
    # so a zero target still leaves noise somewhere to live.
    _A_FLOOR: ClassVar[float] = 1e-3

    # ── Integration resolution ────────────────────────────────────
    # Largest step the *driven* dynamics are integrated with. The
    # oscillator propagation itself is closed-form at any step size
    # (see _propagate), and the PAC modulator is applied as its exact
    # interval mean (see _wave_phase_mean), so the step only needs to
    # resolve the *amplitude* coupling — measured at ~1e-4 for h=0.1 s
    # over a 2 s interval vs the n→∞ limit.
    #
    # 0.1 s is the resolution the neurochemical dynamics were
    # designed at (NeuroTickParams::DEFAULT.dt = 0.1 s). Matching it
    # means the wave and chemical integrators agree on what "one step"
    # means, which is what keeps the two clocks locked together.
    _MAX_SUBSTEP: ClassVar[float] = 0.1

    # Hard ceiling on sub-steps per assess() call, so a multi-second
    # stall cannot cost unbounded CPU. Past this the propagator stays
    # exact at any h; only the coupling terms coarsen.
    _MAX_SUBSTEPS: ClassVar[int] = 64

    # PAC modulation depth — how much slow phase modulates fast
    # amplitude target (fraction of fast band's base target)
    _PAC_DEPTH: ClassVar[float] = 0.15

    # Number of phase bins for modulation index computation (Tort et al., 2010)
    _PAC_BINS: ClassVar[int] = 18

    # ── Regional gamma binding layer ──────────────────────────────
    # Per-region detunings around the gamma carrier (rad/s, ≈±0.5-1.4 Hz
    # around 50 Hz — regional gamma peak dispersion). The spread sets the
    # free-running PLV floor: over the 1 s PLV window a pair detuned by
    # Δω reads |sinc(Δω·T/2)|, so the smallest pairwise difference (6
    # rad/s here) must sweep at least half a cycle per window or
    # *uncoupled* regions still read as locked. At ±1,±3 rad/s the floor
    # measured ≈0.58 — above the 0.5 access threshold, so the PLV could
    # never report "not ignited". At ±3,±9 every pair difference (6, 12,
    # 18 rad/s) is past the first sinc zero and the floor drops to ≈0.05.
    _REGIONAL_GAMMA_DETUNE: ClassVar[dict[str, float]] = {
        "prefrontal": -9.0,
        "parietal": -3.0,
        "temporal": 3.0,
        "occipital": 9.0,
    }
    # Kuramoto pull strength (s^-1) at spectrum-dominant gamma and full
    # plasticity. A region locks to the carrier when the pull exceeds its
    # |detune|, so locking all four needs K > 9 rad/s; the driver is
    # gamma's share of the spectrum (≈0.1 typical wake, ≈0.01 NREM,
    # ≈0.45 gamma-dominant), so the gain must be ≳9/(0.1·0.575)≈160 for a
    # healthy awake cortex to lock hard while NREM stays far below the
    # inner ±3 threshold and the regions free-run — the windowed PLV then
    # collapses on the window's own time (~1 s). At the old 14, K never
    # reached even the inner detunes outside the extreme gamma-dominant
    # edge, so nothing was ever measurably locked.
    _GAMMA_COUPLING_GAIN: ClassVar[float] = 160.0
    # Integration/sampling resolution of the regional layer.
    _REGIONAL_SAMPLE_DT: ClassVar[float] = 0.005
    # Rolling window the PLV is computed over — ~50 gamma cycles.
    _PLV_WINDOW_S: ClassVar[float] = 1.0

    # ── Trace metering ────────────────────────────────────────────
    # Each band emits its real-axis oscillation into an analysis ring
    # at this fixed metering rate; reported band power is measured
    # off the emitted wave by Goertzel (the body's field math), not
    # read from state. 384 Hz keeps epsilon (150 Hz) inside Nyquist
    # where the body's own ~250 Hz probe cannot follow it, and the
    # 1536-sample window is four seconds — bin width 0.25 Hz, so
    # every band center (all multiples of 0.25 Hz) lands exactly on
    # a bin: k = 4f for all ten bands, no off-bin rolloff at all.
    _TRACE_FS: ClassVar[float] = 384.0
    _TRACE_DT: ClassVar[float] = 1.0 / 384.0
    _TRACE_LEN: ClassVar[int] = 1536

    def __init__(
        self,
        rng: random.Random | None = None,
        texture: float = 0.0,
    ) -> None:
        """Initialize the brain wave oscillator.

        Args:
            rng: Source for the stochastic drive. ``None`` seeds from
                system entropy — the noise term is genuine process
                noise, not a scripted wobble.
            texture: Fractional envelope wandering η. ``0.0`` (the
                default) is the deterministic core — the dynamics are
                pinned to their closed form. The production singleton
                opts into ``_TEXTURE``.
        """
        # Quadrature state per band: z = x + iy rotating at the band's
        # own ω. This IS the oscillation — amplitude and phase are
        # measured from it through live views, so there is no
        # separate phase/amplitude state to drift out of sync.
        self._z: dict[BrainWave, complex] = {band: complex(0.2, 0.0) for band in BrainWave}
        self._phase_view = _PhaseView(self._z)
        self._amplitude_view = _AmplitudeView(self._z)
        # The emitted real-axis trace per band — what the Goertzel
        # meter measures. Filled by _emit_trace at _TRACE_FS; the
        # fractional remainder carries forward so emission rate never
        # drifts with sub-step cadence.
        self._trace: dict[BrainWave, deque[float]] = {
            band: deque(maxlen=self._TRACE_LEN) for band in BrainWave
        }
        self._trace_debt: dict[BrainWave, float] = dict.fromkeys(BrainWave, 0.0)
        # The oscillator is a singleton state machine shared by the
        # heartbeat and any reader thread (mind.brain_waves() calls
        # assess too) — a read that iterates the trace while a
        # concurrent assess emits into it is a hard fault, not a
        # stale value. One re-entrant lock serializes the advance and
        # the mutators that feed it.
        self._lock = threading.RLock()
        self._rng = rng if rng is not None else random.Random()
        self._texture = max(0.0, texture)
        self._top_down: dict[BrainWave, float] = dict.fromkeys(BrainWave, 0.0)
        # Two stimulus channels sharing one decay but writing separate
        # registers: ``_stimulus`` leaky-integrates discrete events
        # (pulses); ``_held`` carries a continuous channel's standing
        # level, replaced outright on each report. Splitting them is
        # what lets a pulse survive a held level on the same band —
        # a shared register silently clobbered whichever wrote first
        # (a Berger-effect alpha block from vision died on the field
        # probe's next hold).
        self._stimulus: dict[BrainWave, float] = dict.fromkeys(BrainWave, 0.0)
        self._held: dict[BrainWave, float] = dict.fromkeys(BrainWave, 0.0)
        self._last_target_dominant: BrainWave | None = None
        self._last_phase_name: str | None = None
        self._last_sleep_stage: SleepStage | None = None
        self._last_time: float | None = None
        self._tick_count: int = 0
        # Regional gamma binding layer — the four cortical oscillators
        # the true PLV is measured over. Phases start a quarter-turn
        # apart (PLV = 0): synchrony is earned by the coupling, never
        # assumed. ``_regional_gamma_hist`` is the rolling one-second
        # phase-sample window; ``_regional_gamma_t`` is the layer's own
        # integrated clock so the window stays dt-exact.
        self._regional_gamma_phase: dict[str, float] = dict(
            zip(
                _GAMMA_REGIONS,
                (0.0, math.pi / 2.0, math.pi, 3.0 * math.pi / 2.0),
                strict=True,
            )
        )
        self._regional_gamma_hist: deque[tuple[float, tuple[float, float, float, float]]] = deque()
        self._regional_gamma_t: float = 0.0

    def add_stimulus(self, band: BrainWave, strength: float, decay: float = 0.95) -> None:
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
        with self._lock:
            self._stimulus[band] = self._stimulus[band] * decay + strength

    def hold_stimulus(self, band: BrainWave, level: float) -> None:
        """Set the standing stimulus level for a continuous channel.

        The counterpart of :meth:`add_stimulus` for channels that
        report a *current level* each assessment — substrate energy,
        acoustic emission — rather than discrete events. The level
        is assigned, not integrated: re-reporting the same reading
        holds the stimulus where it is, so the readout is
        independent of how often callers poll. A channel that stops
        reporting simply stops refreshing; the τ≈5 s decay in
        :meth:`_relax_once` fades what remains.

        ``add_stimulus``'s per-call leaky sum is correct for pulse
        injections; applied to a level channel it integrates the
        reading once per call — the standing stimulus then scales
        with the assess rate and machine slow-band energy swamps
        the modeled EEG (a vigilant body reading delta-dominant).

        The level lands in ``_held`` — a separate register from the
        pulse channel — so re-reporting replaces only the standing
        level and never erases a pulse still ringing on the band.
        """
        with self._lock:
            self._held[band] = max(-0.5, min(0.5, level))

    def add_top_down_drive(self, band: BrainWave, strength: float) -> None:
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
        with self._lock:
            self._top_down[band] = min(1.0, self._top_down[band] + max(0.0, strength))

    def perturb_phase(self, band: BrainWave, delta: float) -> None:
        """Nudge a band's instantaneous phase by ``delta`` radians.

        The phase counterpart of :meth:`add_stimulus`: stimulus shifts
        *amplitude*; this shifts *phase*. Physical transients — a
        scheduling shudder, a loud acoustic onset, a power event —
        evoked-reset ongoing oscillations in real EEG; this is the
        model's version of that channel, implemented as what it
        physically is: a rotation of the oscillator's position in its
        own cycle. A positive delta advances the band, negative
        retards it.
        """
        with self._lock:
            self._z[band] *= complex(math.cos(delta), math.sin(delta))

    def reset_phase(self, band: BrainWave, phase: float = 0.0) -> None:
        """Snap a band's phase to ``phase`` (radians).

        A strong transient doesn't nudge the rhythm, it *resets* it —
        the evoked-potential signature where ongoing oscillations
        re-align to the event. The oscillator keeps its amplitude and
        is repositioned at the given phase of its cycle. Use for
        transients large enough to dominate the band's own precession.
        """
        with self._lock:
            r = abs(self._z[band])
            self._z[band] = complex(r * math.cos(phase), r * math.sin(phase))

    @property
    def _phase(self) -> _PhaseView:
        """Measured phase per band — a live view over ``_z``."""
        return self._phase_view

    @_phase.setter
    def _phase(self, values: Any) -> None:
        for band, phase in values.items():
            self._phase_view[band] = phase

    @property
    def _amplitude(self) -> _AmplitudeView:
        """Measured amplitude per band — a live view over ``_z``."""
        return self._amplitude_view

    @_amplitude.setter
    def _amplitude(self, values: Any) -> None:
        for band, amp in values.items():
            self._amplitude_view[band] = amp

    def phase_of(self, band: BrainWave) -> float:
        """Current instantaneous phase of ``band`` — measured, arg z."""
        return self._phase[band]

    def signal_of(self, band: BrainWave) -> float:
        """The band's live trace value, Re(z) — the waveform itself."""
        return self._z[band].real

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
        with self._lock:
            return self._assess_locked(summary, sleep_stage)

    def _assess_locked(
        self,
        summary: NeuroSummary,
        sleep_stage: SleepStage | None,
    ) -> BrainWaveState:
        # Compute targets from neurochemistry (same classification
        # logic as before — these are the neurochemically-driven
        # equilibrium amplitudes)
        targets, label, description = self._compute_targets(summary, sleep_stage)

        # Apply continuous stimulus to targets — held levels plus
        # still-ringing pulses on the same band combine.
        for band in BrainWave:
            stimulus = self._held[band] + self._stimulus[band]
            if stimulus != 0.0:
                targets[band] = max(0.0, targets[band] + stimulus)

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
        dt = self._tick_dt()

        if should_snap:
            self._snap_to_targets(dt, targets)
        else:
            # Propagate the oscillators under the driven targets. Phase
            # and amplitude advance together inside each sub-step —
            # the PAC loop samples the slow bands' measured phase at
            # sub-step resolution, not a precomputed ramp.
            self._advance_relaxation(dt, targets)

        self._last_phase_name = phase_name
        self._last_target_dominant = target_dominant
        self._tick_count += 1

        # Regional gamma binding layer — pulled toward the
        # just-advanced carrier phase on the same dt, so the phase
        # history (and the PLV measured from it) shares the band's
        # own clock.
        self._advance_regional_gamma(dt, summary.plasticity_gate)

        # Build the state from current amplitudes
        return self._build_state(targets, label, description, summary)

    def _tick_dt(self) -> float:
        """Real-time bookkeeping: ``_last_time`` tracks the last
        assessment, not the last relaxation, so the next dt spans
        exactly one interval even across transitions."""
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
            # infraslow gating, and sleep-stage timing.
            #
            # The cap is now a stall guard rather than a frame
            # boundary, and everything inside it is integrated exactly
            # (see _advance_phase_exact / _advance_relaxation).
            dt = min(DT_MAX, max(0.0, elapsed))
        else:
            dt = 0.1  # default 100ms
        self._last_time = now
        return dt

    def _snap_to_targets(self, dt: float, targets: dict[BrainWave, float]) -> None:
        """Re-seat each oscillator's envelope at the new regime's radius
        on a phase/dominant transition, honouring accumulated top-down
        drive: cognition's boost (attention → gamma, filtering →
        alpha, retrieval → theta) takes effect immediately instead of
        being silently discarded until the first driven tick. The drive
        decays here exactly as _relax_once would decay it, so it cannot
        pile up unseen across a run of snaps and then dump all at once.

        The transition is a regime change — the drive changed — so the
        physical move is to rescale the oscillation to its new radius
        *while keeping its phase*: the rhythm carries over rather than
        teleporting. A band with no oscillation yet starts on the
        real axis."""
        # Phase precesses continuously — it must not stall on
        # transition ticks (the relaxed path rotates inside each
        # sub-step; this is the snap-tick counterpart).
        self._advance_phase_exact(dt)
        for band in BrainWave:
            target = targets[band] + self._top_down[band]
            z = self._z[band]
            r = abs(z)
            z = z * (target / r) if r > 0.0 else complex(target, 0.0)
            # The transition arrives textured: the elapsed interval's
            # stochastic drive applies here too, not only on driven
            # ticks — otherwise a regime change lands suspiciously
            # noise-free.
            self._z[band] = self._kick(band, z, dt, target)
            # The snap re-seats the wave's radius for its whole
            # record too: the ring is the meter's memory of *this*
            # oscillation, so its older samples rescale by the same
            # real-positive factor — phase content untouched —
            # or the meter keeps reporting the regime the snap ended.
            ring = self._trace[band]
            if r > 0.0 and len(ring) > 0:
                ratio = abs(self._z[band]) / r
                for i in range(len(ring)):
                    ring[i] *= ratio
            self._emit_trace(band, dt)
        decay = math.exp(-dt / self._DRIVE_TAU)
        for band in BrainWave:
            self._top_down[band] *= decay
        stim_decay = math.exp(-dt / 5.0)
        for band in BrainWave:
            self._stimulus[band] *= stim_decay
            self._held[band] *= stim_decay

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
            powers, label, description = _sleep_phase_powers(summary, sleep_stage)
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
        """Advance the whole bank by dt seconds, single step.

        The unsplit primitive — one dynamics sub-step covering
        rotation, radial growth, noise, and the input decays. Tests
        drive this directly with a dt small enough that no
        sub-stepping is wanted; production code goes through
        :meth:`_advance_relaxation`, which splits the interval so PAC
        samples the slow bands' phase at sub-step resolution.
        """
        self._relax_once(dt, targets)

    def _advance_phase(self, dt: float) -> None:
        """Advance every band's oscillation by dt without retargeting.

        Retained as the stable name for phase advancement; delegates to
        the closed-form rotation, then emits the interval's trace so
        the meter's record covers the span.
        """
        self._advance_phase_exact(dt)
        for band in BrainWave:
            self._emit_trace(band, dt)

    def _emit_trace(self, band: BrainWave, h: float) -> None:
        """Emit the band's shaped trace into the metering ring.

        The wave the Goertzel meter reads: the end phasor gives the
        interval's amplitude (|z|) and end phase (arg z − ωh at the
        start), and each emitted sample is ``r · carrier(φ)`` — the
        band's physiological waveform, not a bare sinusoid. The
        carrier is :func:`_wave_phase_value`, the same table the PAC
        modulator uses, so REM theta emits its sawtooth and the
        meter honestly reads only the fundamental (~2/π of a sine
        at equal amplitude) while the recorded wave carries the
        shape's morphology and harmonics. Emission debt accumulates
        fractionally so the sample count stays proportional to
        elapsed time at any sub-step cadence.
        """
        self._trace_debt[band] += h * self._TRACE_FS
        m = int(self._trace_debt[band])
        self._trace_debt[band] -= m
        if m <= 0:
            return
        freq = self._FREQS[band]
        z = self._z[band]
        r = abs(z)
        phase = math.atan2(z.imag, z.real) - 2.0 * math.pi * freq * h
        step = 2.0 * math.pi * freq * self._TRACE_DT
        two_pi = 2.0 * math.pi
        phase_name = self._last_phase_name or ""
        ring = self._trace[band]
        for _ in range(m):
            ring.append(r * _wave_phase_value(band, phase % two_pi, phase_name))
            phase += step

    def _propagate(self, band: BrainWave, h: float, radius: float) -> None:
        """One step of a band's driven phasor state.

        Rotation is closed form — z ← z·e^{iωh}, the angle reduced to
        its fractional turn first and a within-noise integral-cycle
        residue snapped to zero (see :meth:`_advance_phase_exact`).
        The envelope relaxes exponentially toward the driven radius —
        r ← r + (a − r)·(1 − e^{−h/τ}), a plain first-order approach:
        the limit-cycle map moved to the body's substrate resonator.
        The kick is additive — z ← z + σ√h·(g₁ + ig₂), with
        σ = √2·η·(A + floor)/√τ chosen so the radial OU process
        reaches stationary std ≈ η·A: the envelope wanders that
        fraction about its radius regardless of the assess cadence
        (kick variance sums ∝ dt, so cadence splits are equal in
        distribution). At the end of the step the band emits its
        real-axis trace — the signal the Goertzel meter reads.
        """
        z = self._z[band]
        if h > 0.0:
            # Rotation — closed form, fractional-turn reduction.
            turns = self._FREQS[band] * h
            nearest = round(turns)
            frac = 0.0 if abs(turns - nearest) <= _TURN_SNAP_EPS else turns - nearest
            ang = 2.0 * math.pi * frac
            z *= complex(math.cos(ang), math.sin(ang))

            # Radial relaxation toward the driven radius — first-order
            # exponential approach on the same per-band τ the
            # limit-cycle map used.
            a = max(radius, self._A_FLOOR)
            u = z.real * z.real + z.imag * z.imag
            if u > 0.0:
                r = math.sqrt(u)
                z *= (r + (a - r) * (1.0 - math.exp(-h / self._TAU[band]))) / r

            # Stochastic drive — variance ∝ h, so cadence-invariant.
            z = self._kick(band, z, h, a)

            # Seat the end state before the trace reads it — the
            # meter must see the wave this step produced, not the
            # one it replaced.
            self._z[band] = z
            self._emit_trace(band, h)

        else:
            self._z[band] = z

    def _kick(self, band: BrainWave, z: complex, h: float, radius: float) -> complex:
        """One stochastic kick: z + σ√h·(g₁ + ig₂).

        σ = √2·η·(A + floor)/√τ is chosen so the radial Ornstein–
        Uhlenbeck process reaches stationary std ≈ η·A — the envelope
        wanders that fraction about its driven radius regardless of
        the assess cadence (kick variance sums ∝ elapsed time, so any
        subdivision of an interval is equal in distribution). Used by
        both the driven propagator and the transition snap — a regime
        change should arrive already textured, not noise-free.
        """
        if self._texture <= 0.0 or h <= 0.0:
            return z
        s = (
            math.sqrt(2.0)
            * self._texture
            * (max(radius, 0.0) + self._NOISE_FLOOR)
            * math.sqrt(h / self._TAU[band])
        )
        return z + complex(self._rng.gauss(0.0, s), self._rng.gauss(0.0, s))

    def _advance_phase_exact(self, dt: float) -> None:
        """Rotate every band's state by ω·dt in closed form.

        The phase part of the propagator on its own: each band's
        quadrature state obeys ż = iωz for the rotation, so the exact
        solution over any interval is z ← z·e^{iωdt}. There is no
        accumulation error and no dependence on how the interval was
        subdivided — rotating 5 s in one call and as fifty 0.1 s steps
        land on the same phase (rotation composes exactly).

        The angle is reduced to its fractional turn *before* the trig,
        then a within-noise integral-cycle residue is snapped to zero.
        For a 2 s step at THETA's 5.5 Hz the raw count is 11 turns but
        5.5·2 evaluates to 10.999999999999998 — subtracting the nearest
        turn leaves ~one full turn the wrong way, so the residue must
        be snapped, not integrated. This is a precision tidy-up (the
        drift is ~1e-14 turns per step either way) pinned by
        test_integral_cycle_counts_return_exactly_to_start.
        """
        if dt <= 0.0:
            return
        two_pi = 2.0 * math.pi
        for band in BrainWave:
            turns = self._FREQS[band] * dt
            nearest = round(turns)
            frac = 0.0 if abs(turns - nearest) <= _TURN_SNAP_EPS else turns - nearest
            ang = two_pi * frac
            self._z[band] *= complex(math.cos(ang), math.sin(ang))

    def _advance_regional_gamma(self, dt: float, plasticity: float) -> None:
        """Advance the four regional gamma oscillators and sample them.

        Each region precesses at the carrier's angular frequency plus a
        small fixed detuning and is pulled toward the global gamma
        phase by a Kuramoto coupling term. The coupling scales with the
        band's normalised amplitude and the plasticity gate — a
        long-range cortico-cortical lock needs both something to lock
        to (gamma power) and the synaptic capacity to hold it, which is
        how stress and sleep dissolve binding *dynamically* rather than
        by a rule that says so. Phases are integrated at 5 ms and kept
        in a rolling one-second window; ``compute_gamma_synchrony``
        reads the windowed pairwise PLV off the state.
        """
        if dt <= 0.0:
            return
        omega = 2.0 * math.pi * self._FREQS[BrainWave.GAMMA]
        total = sum(self._amplitude.values())
        gamma_amp = self._amplitude[BrainWave.GAMMA] / total if total > 0.0 else 0.0
        coupling = self._GAMMA_COUPLING_GAIN * gamma_amp * (0.15 + 0.85 * plasticity)
        # The carrier rotation is ω·dt per step (the noise kick adds
        # only sub-radian phase diffusion), so the phase inside the
        # interval interpolates backward from the measured end phase.
        gamma_end = self._phase[BrainWave.GAMMA]
        steps = max(1, int(dt / self._REGIONAL_SAMPLE_DT))
        h = dt / steps
        two_pi = 2.0 * math.pi
        for s in range(1, steps + 1):
            t_rel = s * h
            phi_g = gamma_end - omega * (dt - t_rel)
            for region in _GAMMA_REGIONS:
                omega_r = omega + self._REGIONAL_GAMMA_DETUNE[region]
                pull = coupling * math.sin(phi_g - self._regional_gamma_phase[region])
                self._regional_gamma_phase[region] = (
                    self._regional_gamma_phase[region] + (omega_r + pull) * h
                ) % two_pi
            self._regional_gamma_hist.append(
                (
                    self._regional_gamma_t + t_rel,
                    (
                        self._regional_gamma_phase["prefrontal"],
                        self._regional_gamma_phase["parietal"],
                        self._regional_gamma_phase["temporal"],
                        self._regional_gamma_phase["occipital"],
                    ),
                )
            )
        self._regional_gamma_t += dt
        cutoff = self._regional_gamma_t - self._PLV_WINDOW_S
        while self._regional_gamma_hist and self._regional_gamma_hist[0][0] < cutoff:
            self._regional_gamma_hist.popleft()

    def _advance_relaxation(
        self,
        dt: float,
        targets: dict[BrainWave, float],
    ) -> None:
        """Advance the oscillator bank over dt seconds in sub-steps.

        Each sub-step propagates every band's phasor state
        (rotation + exponential radial approach + noise kick) toward
        its PAC-modulated driven radius, and decays the top-down
        drive and stimulus registers.

        Sub-stepping exists for the *coupled* terms, not integrator
        accuracy — the propagator is closed-form at any step size, the
        PAC modulator is applied as its exact interval mean, and the
        two decays compose exactly
        (``exp(-(h1+h2)/τ) == exp(-h1/τ)·exp(-h2/τ)``). What fine steps
        resolve is the amplitude-coupled PAC strength (the geometric
        mean of the bands' envelopes, read at each step's start) —
        measured residual ~2e-4 at the 0.1 s resolution.

        Args:
            dt: Total interval in seconds. Assumed non-negative.
            targets: Neurochemically-driven equilibrium radii.
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
        """One dynamics sub-step: PAC-modulated radii, propagate, decay.

        Rotation and radial growth happen inside :meth:`_propagate` —
        phase is measured oscillator state now, so it advances *with*
        each sub-step, and the PAC loop below applies the slow bands'
        carrier as its exact interval mean (:func:`_wave_phase_mean`).
        """
        # 1. Cross-frequency phase-amplitude modulation
        # Slow band measured phase modulates fast band cycle radius
        modulated_targets = dict(targets)
        slow_bands = (
            BrainWave.INFRASLOW,
            BrainWave.DELTA,
            BrainWave.THETA,
            BrainWave.ALPHA,
            BrainWave.SIGMA,
        )
        fast_bands = (
            BrainWave.BETA1,
            BrainWave.BETA2,
            BrainWave.BETA3,
            BrainWave.GAMMA,
            BrainWave.EPSILON,
        )
        # REM alpha intermittency is driven by the infraslow band:
        # alpha flashes wax/wane on the infraslow phase, so REM
        # is not bathed in sustained alpha. Infraslow is ~0.25 Hz, so
        # its phase barely moves inside a step — a midpoint sample is
        # exact to ~1e-4.
        if self._last_phase_name == "rem":
            infra_mid = self._phase[BrainWave.INFRASLOW] + (
                math.pi * self._FREQS[BrainWave.INFRASLOW] * dt
            )
            modulated_targets[BrainWave.ALPHA] *= _rem_alpha_envelope(infra_mid)
        for slow in slow_bands:
            for fast in fast_bands:
                if slow == fast:
                    continue
                # PAC strength: geometric mean of measured amplitudes
                pac = math.sqrt(self._amplitude[slow] * self._amplitude[fast])
                # Phase modulation: the *mean* of the slow band's
                # carrier over this step (sawtooth for REM theta,
                # cosine otherwise) ∈ [-1, 1] — the exact average the
                # fast oscillator felt, not a start-of-step sample.
                # Positive → enlarge fast cycle, negative → shrink it
                phase_mod = _wave_phase_mean(
                    slow,
                    self._phase[slow],
                    2.0 * math.pi * self._FREQS[slow] * dt,
                    self._last_phase_name or "",
                )
                modulated_targets[fast] += (
                    pac * phase_mod * self._PAC_DEPTH * modulated_targets[fast]
                )

        # 2. Propagate each oscillator toward its modulated radius
        for band in BrainWave:
            radius = max(0.0, modulated_targets[band])
            # Top-down drive enlarges the cycle above the neurochemical
            # radius
            radius += self._top_down[band]
            self._propagate(band, dt, radius)

        # 3. Decay top-down drive
        decay = math.exp(-dt / self._DRIVE_TAU)
        for band in BrainWave:
            self._top_down[band] *= decay

        # 4. Decay continuous stimulus (τ ≈ 5s for slow decay) —
        # both registers: held levels fade when a channel goes
        # silent, pulses ring down on the same constant.
        stim_decay = math.exp(-dt / 5.0)
        for band in BrainWave:
            self._stimulus[band] *= stim_decay
            self._held[band] *= stim_decay

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
        """Build BrainWaveState from the measured band powers."""
        # Band power is *measured*: Goertzel over each band's emitted
        # real-axis trace — the field probe's own meter moved into the
        # mind — not read from oscillator amplitude. The meter reports
        # the single-bin amplitude estimate (2√P = A for an on-bin
        # tone), the same scale "powers" has always carried and every
        # downstream threshold is tuned to. A band's meter reports only
        # once the ring's bin nearest its center actually resolves it —
        # |k·fs/n − f| ≤ f/8 — a meter that would measure a theta wave
        # at an alpha bin should not report at all; until the span is
        # honest the state amplitude stands in. Fast bands resolve
        # early, slow bands need their span: infraslow only at the full
        # four seconds.
        powers = {}
        for band in BrainWave:
            ring = self._trace[band]
            n = len(ring)
            freq = self._FREQS[band]
            if n >= 4:
                k = max(1, round(n * freq / self._TRACE_FS))
                resolves = abs(k * self._TRACE_FS / n - freq) <= freq / 8.0
            else:
                resolves = False
            if resolves:
                # Single-bin amplitude estimate: for x = A·cos, the
                # Goertzel bin has |X| = nA/2, so 2√(|X|²/n²) = A.
                # Powers stay amplitude-proportional — the same scale
                # the field has always reported and every downstream
                # threshold is tuned to.
                powers[band] = 2.0 * math.sqrt(
                    _band_power(ring, self._TRACE_FS, freq)
                )
            else:
                powers[band] = self._amplitude[band]
        total = sum(powers.values())
        if total > 0:
            for band in powers:
                powers[band] /= total

        # Find dominant and secondary
        sorted_waves = sorted(powers.items(), key=lambda x: x[1], reverse=True)
        dominant = sorted_waves[0][0]
        secondary = sorted_waves[1][0]

        # Derive cognitive gating parameters
        plasticity = summary.plasticity_gate
        consolidation_w = summary.consolidation_weight
        encoding_w = summary.encoding_weight
        focus = _compute_focus(powers)
        integration = _compute_integration(powers, plasticity)
        consolidation = _compute_consolidation(powers, consolidation_w, encoding_w)

        gamma_plv, gamma_plv_pairs = _windowed_gamma_plv(self._regional_gamma_hist)
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
            regional_gamma_phases=dict(self._regional_gamma_phase),
            gamma_plv=gamma_plv,
            gamma_plv_pairs=gamma_plv_pairs,
        )
        # Cross-frequency coupling from actual phase state
        state.cross_frequency = self._compute_cross_frequency(powers, consolidation_w, encoding_w)
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
            BrainWave.INFRASLOW,
            BrainWave.DELTA,
            BrainWave.THETA,
            BrainWave.ALPHA,
            BrainWave.SIGMA,
        )
        fast_bands = (
            BrainWave.BETA1,
            BrainWave.BETA2,
            BrainWave.BETA3,
            BrainWave.GAMMA,
            BrainWave.EPSILON,
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
                mode = _cognitive_mode_for_coupling(slow.value, fast.value)
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
        _oscillator = BrainWaveOscillator(texture=BrainWaveOscillator._TEXTURE)
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


def hold_stimulus(band: BrainWave, level: float) -> None:
    """Hold a continuous channel's standing stimulus level.

    For channels reporting a current level each assessment —
    physical substrate energy, acoustic emission — where the
    reading *is* the stimulus, not an event to integrate. See
    :meth:`BrainWaveOscillator.hold_stimulus`.
    """
    _get_oscillator().hold_stimulus(band, level)


def perturb_wave_phase(band: BrainWave, delta: float) -> None:
    """Nudge a band's phase — the evoked-response channel.

    A physical transient shifts ongoing rhythm phase rather than
    only adding amplitude; this is how real events write into the
    oscillator's timing, not just its power.
    """
    _get_oscillator().perturb_phase(band, delta)


def reset_wave_phase(band: BrainWave, phase: float = 0.0) -> None:
    """Snap a band's phase — the strong-transient reset."""
    _get_oscillator().reset_phase(band, phase)


def wave_phase_of(band: BrainWave) -> float:
    """Current instantaneous phase of ``band`` (radians)."""
    return _get_oscillator().phase_of(band)


def reset_oscillator(texture: float = 0.0) -> None:
    """Reset the oscillator to initial state.

    Used by tests to ensure a clean state on the deterministic core
    (no stochastic drive by default — pass ``texture`` to include
    it). In normal operation the oscillator persists across calls
    and accumulates temporal dynamics under its noise drive.
    """
    global _oscillator
    _oscillator = BrainWaveOscillator(texture=texture)


def assess_brain_waves(
    summary: NeuroSummary,
    sleep_stage: SleepStage | None = None,
    client: Any | None = None,
) -> BrainWaveState:
    """Derive brain wave state from the coupled oscillator.

    Pass ``sleep_stage`` from the ultradian sleep-cycle tracker when
    asleep so the waves match the stage the rest of the sleep system
    is acting on (see :func:`_sleep_phase_powers`).

    When ``client`` is supplied, the daemon's field-probe signature —
    the body's *real* measured substrate dynamics — is applied as
    stimulus before the assess step, so the wave state carries actual
    physics, not only neurochemical targets (see
    ``brainstem.field_probe``). With no client the oscillator runs on
    its modeled inputs alone — identical to the old call shape.

    The oscillator propagates each band's phasor state —
    rotating, relaxing toward the neurochemically-driven radius,
    wandering under its noise drive — then returns the measured
    state. The mapping from neurochemistry to driven
    radii is grounded in neuroscience:
    - Alertness is the primary driver (low→delta/theta, high→beta/gamma)
    - Plasticity gates whether high alertness produces gamma (integration)
      or just beta (alert but not integrating)
    - Phase overrides: sleep → delta/theta, flow → gamma
    - Consolidation weight drives theta power

    On phase transitions or dominant-band changes, oscillations are
    re-seated at the new regime's radius while keeping their phase.
    Within a stable state the envelopes converge smoothly with
    band-specific time constants, producing genuine temporal
    dynamics. Top-down drive from cognition/action/perception can
    enlarge specific cycles above their neurochemical radii.
    """
    if client is not None:
        from brainstem.field_probe import apply_field_stimulus

        apply_field_stimulus(client)
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
            BrainWave.INFRASLOW: 0.08,
            _d: delta,
            _t: theta,
            _a: alpha,
            BrainWave.SIGMA: 0.0,
            BrainWave.BETA1: 0.03,
            BrainWave.BETA2: 0.03,
            BrainWave.BETA3: 0.02,
            BrainWave.GAMMA: 0.02,
            BrainWave.EPSILON: 0.0,
        }

    anchor_high = _anchor(0.35, 0.32, 0.06)  # cortical arousal 0.50
    anchor_mid = _anchor(0.12, 0.45, 0.22)  # 0.30
    anchor_low = _anchor(0.08, 0.25, 0.45)  # 0.15

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
            "delta rising over theta — the descent has gone past drowsy into slow-wave territory"
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
            BrainWave.INFRASLOW: 0.02,
            BrainWave.DELTA: 0.05,
            BrainWave.THETA: 0.10,
            BrainWave.ALPHA: 0.10,
            BrainWave.SIGMA: 0.00,
            BrainWave.BETA1: 0.05,
            BrainWave.BETA2: 0.15,
            BrainWave.BETA3: 0.35,
            BrainWave.GAMMA: 0.08,
            BrainWave.EPSILON: 0.00,
        }
        label = "beta3"
        description = "too much input, I can't process it all"
    elif phase == "flow":
        # Flow: gamma dominant — everything integrating effortlessly
        powers = {
            BrainWave.INFRASLOW: 0.01,
            BrainWave.DELTA: 0.02,
            BrainWave.THETA: 0.08,
            BrainWave.ALPHA: 0.10,
            BrainWave.SIGMA: 0.00,
            BrainWave.BETA1: 0.03,
            BrainWave.BETA2: 0.12,
            BrainWave.BETA3: 0.05,
            BrainWave.GAMMA: 0.45,
            BrainWave.EPSILON: 0.03,
        }
        label = "gamma"
        description = "everything clicking, distant ideas connecting"
    else:
        # Stress: high beta dominant, gamma strongly suppressed — alert but
        # rigid. Stress produces hypervigilant high beta with impaired
        # gamma-synchronized integration (Herrmann & Demiralp, 2012).
        powers = {
            BrainWave.INFRASLOW: 0.02,
            BrainWave.DELTA: 0.05,
            BrainWave.THETA: 0.08,
            BrainWave.ALPHA: 0.08,
            BrainWave.SIGMA: 0.00,
            BrainWave.BETA1: 0.04,
            BrainWave.BETA2: 0.15,
            BrainWave.BETA3: 0.35,
            BrainWave.GAMMA: 0.07,
            BrainWave.EPSILON: 0.00,
        }
        label = "beta3"
        description = "tense and alert, can't see the big picture"
    return powers, label, description


def _sleep_phase_powers(
    summary: NeuroSummary,
    stage: SleepStage | None = None,
) -> tuple[dict[BrainWave, float], str, str]:
    """Derive brain wave powers for sleep stages (REM/N1/N2/N3).

    ``stage`` is the stage the daemon's sleep-wake network reports, with
    NREM depth from the network's sleep pressure (see
    ``_sleep_stage_from_summary``). When it is None the stage is derived
    from the summary instead, using the arousal bands, which cannot express
    NREM depth now that the network sets NREM arousal from LC alone.
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
            BrainWave.INFRASLOW: 0.03,
            BrainWave.DELTA: 0.06,
            BrainWave.THETA: 0.40,
            BrainWave.ALPHA: 0.10,
            BrainWave.SIGMA: 0.00,
            BrainWave.BETA1: 0.06,
            BrainWave.BETA2: 0.16,
            BrainWave.BETA3: 0.07,
            BrainWave.GAMMA: 0.16,
            BrainWave.EPSILON: 0.00,
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
            BrainWave.INFRASLOW: 0.05,
            BrainWave.DELTA: 0.15,
            BrainWave.THETA: 0.35,
            BrainWave.ALPHA: 0.22,
            BrainWave.SIGMA: 0.05,
            BrainWave.BETA1: 0.04,
            BrainWave.BETA2: 0.04,
            BrainWave.BETA3: 0.03,
            BrainWave.GAMMA: 0.03,
            BrainWave.EPSILON: 0.00,
        }
        label = "theta"
        description = "light sleep, drifting between wake and sleep"
    elif stage is SleepStage.N2:
        # N2: theta/delta background punctuated by sleep spindles
        # (12-16 Hz sigma, modelled here as elevated sigma band)
        # and K-complexes. The workhorse of memory consolidation.
        powers = {
            BrainWave.INFRASLOW: 0.05,
            BrainWave.DELTA: 0.20,
            BrainWave.THETA: 0.25,
            BrainWave.ALPHA: 0.12,
            BrainWave.SIGMA: 0.20,
            BrainWave.BETA1: 0.05,
            BrainWave.BETA2: 0.05,
            BrainWave.BETA3: 0.03,
            BrainWave.GAMMA: 0.03,
            BrainWave.EPSILON: 0.00,
        }
        label = "sigma"
        description = "N2 sleep — spindles and K-complexes, memories transferring to neocortex"
    else:
        # N3: slow-wave sleep — delta-dominant (0.5-4 Hz), the
        # deepest, most restorative stage. Sharp wave-ripples,
        # glymphatic clearance.
        powers = {
            BrainWave.INFRASLOW: 0.05,
            BrainWave.DELTA: 0.50,
            BrainWave.THETA: 0.25,
            BrainWave.ALPHA: 0.08,
            BrainWave.SIGMA: 0.05,
            BrainWave.BETA1: 0.02,
            BrainWave.BETA2: 0.02,
            BrainWave.BETA3: 0.01,
            BrainWave.GAMMA: 0.01,
            BrainWave.EPSILON: 0.02,
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
        powers, label, description = _alert_powers(plasticity)
    elif alertness > 0.45:
        # Moderate alertness — alpha dominant
        # Alpha filters: high alpha = strong filtering
        alpha_power = 0.35 + (0.55 - abs(alertness - 0.55)) * 0.3
        powers = {
            BrainWave.INFRASLOW: 0.05,
            BrainWave.DELTA: 0.08,
            BrainWave.THETA: 0.15,
            BrainWave.ALPHA: alpha_power,
            BrainWave.SIGMA: 0.00,
            BrainWave.BETA1: 0.05,
            BrainWave.BETA2: 0.08,
            BrainWave.BETA3: 0.04,
            BrainWave.GAMMA: 0.05,
            BrainWave.EPSILON: 0.00,
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
            BrainWave.INFRASLOW: 0.05,
            BrainWave.DELTA: 0.08,
            BrainWave.THETA: 0.20 + theta_boost,
            BrainWave.ALPHA: alpha_power,
            BrainWave.SIGMA: 0.00,
            BrainWave.BETA1: 0.04,
            BrainWave.BETA2: 0.05,
            BrainWave.BETA3: 0.03,
            BrainWave.GAMMA: 0.03,
            BrainWave.EPSILON: 0.00,
        }
        # Label follows the dominant band in this borderline band.
        if powers[BrainWave.ALPHA] >= powers[BrainWave.THETA]:
            label = "alpha"
            description = (
                "alpha holding with theta creeping in — the wake/N1 border, filtering easing"
            )
        else:
            label = "theta"
            description = "theta overtaking fragmenting alpha — N1 onset, drifting toward sleep"
    else:
        # Alertness <= 0.35 is resolved to the drowsy signature
        # before this branch is reached — deltas down-regulating
        # the arousal register are never reported as "active."
        # Defensive fallback only.
        powers = {
            BrainWave.INFRASLOW: 0.05,
            BrainWave.DELTA: 0.40,
            BrainWave.THETA: 0.25,
            BrainWave.ALPHA: 0.12,
            BrainWave.SIGMA: 0.00,
            BrainWave.BETA1: 0.03,
            BrainWave.BETA2: 0.04,
            BrainWave.BETA3: 0.02,
            BrainWave.GAMMA: 0.02,
            BrainWave.EPSILON: 0.00,
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


def _alert_powers(
    plasticity: float,
) -> tuple[dict[BrainWave, float], str, str]:
    """High alertness — gamma when plasticity supports integration,
    beta otherwise."""
    if plasticity > 0.5:
        # High plasticity → gamma (integration)
        gamma_boost = (plasticity - 0.5) * 0.6
        powers = {
            BrainWave.INFRASLOW: 0.02,
            BrainWave.DELTA: 0.03,
            BrainWave.THETA: 0.07,
            BrainWave.ALPHA: 0.08,
            BrainWave.SIGMA: 0.00,
            BrainWave.BETA1: 0.03,
            BrainWave.BETA2: 0.12,
            BrainWave.BETA3: 0.05,
            BrainWave.GAMMA: 0.40 + gamma_boost * 0.3,
            BrainWave.EPSILON: 0.02,
        }
        return powers, "gamma", "sharp and integrative, ideas are connecting"
    # Low plasticity → just beta (alert but not integrating)
    powers = {
        BrainWave.INFRASLOW: 0.03,
        BrainWave.DELTA: 0.05,
        BrainWave.THETA: 0.08,
        BrainWave.ALPHA: 0.10,
        BrainWave.SIGMA: 0.00,
        BrainWave.BETA1: 0.05,
        BrainWave.BETA2: 0.25,
        BrainWave.BETA3: 0.15,
        BrainWave.GAMMA: 0.15,
        BrainWave.EPSILON: 0.00,
    }
    return powers, "beta2", "alert and focused, but not making new connections"


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
    The neurochemical consolidation_weight gates this, and the
    encoding_weight counter-gates it when encoding dominates.
    """
    theta = powers[BrainWave.THETA]
    delta = powers[BrainWave.DELTA]
    sigma = powers[BrainWave.SIGMA]

    base = theta * 0.9 + delta * 0.6 + sigma * 0.5
    # Gate by the actual consolidation weight from neurochemistry.
    # Encoding and consolidation are opposing modes (the same balance
    # compute_theta_gamma_coupling uses to pick a phase): when the
    # encoding weight outranks the consolidation weight the system is
    # in write mode, not replay mode, and the drive is suppressed in
    # proportion to the encoding advantage.
    gate = 0.5 + consolidation_weight * 0.5 - max(0.0, encoding_weight - consolidation_weight) * 0.5
    consolidation = base * gate

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
    infraslow = powers.get(BrainWave.INFRASLOW, 0.0)

    positive = gamma * 0.45 + beta2 * 0.20 + beta3 * 0.10 + beta1 * 0.05 + alpha * 0.05
    # Theta weighs 0.30 (75% of delta's 0.40): frontal midline theta
    # is memory-consolidation effort — a sleep-direction drive nearly
    # as strong as delta. At 0.20, a deep-sleep state (55% delta +
    # 25% theta) landed at engagement 0.19 and, after the 70/30 body
    # blend, exactly on neutral nice (5) — violating the contract
    # that delta-dominant sleep is strictly below neutral priority
    # (test_brain_waves.py::test_derive_self_priority_delta_dominant).
    negative = theta * 0.30 + delta * 0.40 + infraslow * 0.10
    drive = positive - negative
    min_drive = -0.40  # all delta (primary sleep band)
    max_drive = 0.45  # all gamma
    engagement = (drive - min_drive) / (max_drive - min_drive)
    return max(0.0, min(1.0, engagement))


def _derive_io_class(state: BrainWaveState, engagement: float) -> str:
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
