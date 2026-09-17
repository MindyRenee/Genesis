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

- **Gamma (30-100 Hz)** — information integration, binding disparate
  concepts into a coherent moment. When you "get it," that's gamma.
- **Beta (13-30 Hz)** — active, alert thinking. Problem-solving,
  focused attention, motor decisions.
- **Alpha (8-13 Hz)** — relaxed reflection. Alpha *inhibits* irrelevant
  information, filtering noise so the signal stands out. It's not
  just "calm" — it's active suppression of distraction.
- **Theta (4-8 Hz)** — memory consolidation, emotional processing,
  creativity. REM sleep is theta-dominant. Hippocampal replay happens
  here.
- **Delta (0.5-4 Hz)** — deep restorative sleep. Housekeeping,
  glymphatic clearance, noncognitive maintenance.

# How Genesis's brain waves work

Genesis's brain waves are produced by a **coupled amplitude-phase
oscillator** (``BrainWaveOscillator``), not a classifier. Each band
has persistent phase and amplitude state:

- **Phase** advances at the band's characteristic frequency (delta
  ~2 Hz, theta ~6 Hz, alpha ~10 Hz, beta ~20 Hz, gamma ~40 Hz).
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
  gamma, memory retrieval → theta, motor planning → beta, V1 visual
  input → gamma. This is bidirectional coupling: the oscillator
  gates cognition, and cognition drives the oscillator.

The neurochemical-to-target mapping is grounded in neuroscience:
- High alertness + high plasticity → gamma target (integration)
- High alertness + low plasticity → beta target (alert, not integrating)
- Moderate alertness → alpha target (reflective filtering)
- Low alertness → theta target (memory consolidation)
- Sleep phase → delta/theta target (restorative/dreaming)

# Cognitive effects

The oscillator's amplitude distribution produces three cognitive
gating parameters:

- `focus` — how narrow the attention (0=broad, 1=narrow)
- `integration` — how likely distant concepts connect (0=isolated, 1=bound)
- `consolidation` — how much memory consolidation happens (0=none, 1=maximal)

These gate how the cognition engine processes information:
- **Gamma**: boosts cross-network integration. Distant concepts are
  more likely to connect. Reasoning chains go deeper.
- **Beta**: standard processing. Focused but not integrative.
- **Alpha**: filters peripheral concepts. Only highly-activated
  concepts reach cognition. Reduces noise, increases signal.
- **Theta**: prioritizes memory consolidation and emotional
  processing. Novel connections more likely. Creativity boosted.
- **Delta**: minimal cognitive processing. Housekeeping mode.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, ClassVar

from genesis_client import NeuroSummary

__all__ = [
    "BrainWave",
    "BrainWaveOscillator",
    "BrainWaveState",
    "CrossFrequencyCoupling",
    "GammaSynchrony",
    "SleepStage",
    "SleepStageSignature",
    "ThetaGammaCoupling",
    "add_brain_wave_drive",
    "apply_self_priority",
    "assess_brain_waves",
    "compute_gamma_synchrony",
    "compute_theta_gamma_coupling",
    "derive_self_priority",
    "reset_oscillator",
]


class BrainWave(Enum):
    """The five major brain wave bands."""

    GAMMA = "gamma"  # 30-100 Hz — integration, binding
    BETA = "beta"  # 13-30 Hz — active thinking, alert
    ALPHA = "alpha"  # 8-13 Hz — relaxed reflection, filtering
    THETA = "theta"  # 4-8 Hz — memory, creativity, REM
    DELTA = "delta"  # 0.5-4 Hz — deep sleep, restoration


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
      (10-16 Hz sigma bursts) and K-complexes — the defining
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


# ─── Theta-gamma cross-frequency coupling ──────────────────────


@dataclass(slots=True)
class ThetaGammaCoupling:
    """Theta-gamma cross-frequency coupling (CFC) state.

    Gamma amplitude (~40-100 Hz) is modulated by the phase of the
    theta oscillation (~4-8 Hz). This phase-amplitude coupling (PAC)
    is one of the most robust cross-frequency relationships in the
    brain and is critical for memory:

    - **Encoding**: gamma power peaks near the theta *trough* (or
      falling phase), when hippocampal pyramidal cells are most
      excitable — favouring the formation of new associations
      (Lega et al., 2012; Heusser et al., 2016).
    - **Retrieval**: gamma power peaks near the theta *peak* (or
      rising phase), favouring the reactivation of stored
      representations (Kaplan et al., 2014; Griffiths et al., 2019).

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
    # ~0 = theta peak (retrieval-favouring); ~π = theta trough
    # (encoding-favouring).
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

    The preferred phase — and thus the encoding vs retrieval mode — is
    determined by the encoding/consolidation balance: high encoding
    weight biases toward the theta trough (encoding), high
    consolidation weight biases toward the theta peak (retrieval/
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

    # Preferred phase: encoding bias → theta trough (π, encoding);
    # consolidation bias → theta peak (0, retrieval/replay).
    # Interpolate by the encoding-vs-consolidation balance.
    total = encoding_weight + consolidation_weight
    if total <= 0.0:
        balance = 0.5
    else:
        balance = encoding_weight / total  # 0 = all consolidation, 1 = all encoding
    # balance 0 → phase 0 (peak, retrieval); balance 1 → phase π (trough, encoding)
    preferred_phase = balance * math.pi

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
        "gamma": "integration",
        "beta": "maintenance",
        "alpha": "gating",
    }.get(fast, "coupling")
    # Slow band gives the *state* in which the gating happens
    slow_role = {
        "delta": "rest",
        "theta": "encoding",
        "alpha": "filtering",
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
    """Compute gamma-band synchrony across brain regions.

    Gamma synchrony is modelled from the global gamma power (the
    "ignition" signal) gated by plasticity (which determines whether
    long-range cortico-cortical connections can sustain phase locking
    — chronic stress / low plasticity impairs gamma synchrony,
    matching the stress-gamma reduction in the neurobiology audit;
    Herrmann & Demiralp, 2012).

    Each region receives a gamma power drawn from the global gamma
    power with small region-specific variation. The phase-locking
    value (PLV) across regions is the global gamma power scaled by
    plasticity — high gamma + high plasticity → high PLV → cognitive
    access; low gamma or low plasticity → low PLV → noncognitive.

    Args:
        state: The current brain wave state.
        plasticity: Neurochemical plasticity gate [0,1].

    Returns:
        A ``GammaSynchrony`` with per-region powers, PLV, and a
        cognitive-access prediction.
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
    for region in _GAMMA_REGIONS:
        # Frontal/parietal (workspace hubs) get a slight boost;
        # sensory regions slightly lower.
        hub_boost = 0.1 if region in ("prefrontal", "parietal") else -0.05
        power = max(0.0, min(1.0, gamma + hub_boost * gamma))
        region_powers[region] = power

    # Cognitive access threshold: PLV > 0.5 predicts cognitive access
    # (Dehaene, 2014 — the "ignition" threshold).
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
      (delta ~2 Hz, theta ~6 Hz, alpha ~10 Hz, beta ~20 Hz, gamma
      ~40 Hz). Phase is genuine oscillator state — it precesses
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
      → gamma, memory retrieval → theta, motor activity → beta.
      The drive decays exponentially (τ=2s), so it must be
      sustained by ongoing activity.

    The oscillator is a Wilson-Cowan-style population model
    simplified to amplitude-phase form. It produces genuine
    oscillatory state (frequency, phase, amplitude) from which
    band powers, cross-frequency coupling, and cognitive gating
    parameters are derived.
    """

    # Characteristic frequencies (Hz) — center of each band
    _FREQS: ClassVar[dict[BrainWave, float]] = {
        BrainWave.DELTA: 2.0,    # 0.5-4 Hz
        BrainWave.THETA: 6.0,    # 4-8 Hz
        BrainWave.ALPHA: 10.0,   # 8-13 Hz
        BrainWave.BETA: 20.0,    # 13-30 Hz
        BrainWave.GAMMA: 40.0,   # 30-100 Hz
    }

    # Amplitude relaxation time constants (seconds).
    # Slow bands have inertia; fast bands track quickly.
    _TAU: ClassVar[dict[BrainWave, float]] = {
        BrainWave.DELTA: 8.0,
        BrainWave.THETA: 4.0,
        BrainWave.ALPHA: 2.0,
        BrainWave.BETA: 1.0,
        BrainWave.GAMMA: 0.5,
    }

    # Top-down drive decay time constant (seconds)
    _DRIVE_TAU: ClassVar[float] = 2.0

    # PAC modulation depth — how much slow phase modulates fast
    # amplitude target (fraction of fast band's base target)
    _PAC_DEPTH: ClassVar[float] = 0.15

    def __init__(self) -> None:
        """Initialize the brain wave oscillator."""
        self._phase: dict[BrainWave, float] = dict.fromkeys(BrainWave, 0.0)
        self._amplitude: dict[BrainWave, float] = dict.fromkeys(BrainWave, 0.2)
        self._top_down: dict[BrainWave, float] = dict.fromkeys(BrainWave, 0.0)
        self._last_target_dominant: BrainWave | None = None
        self._last_phase_name: str | None = None
        self._last_time: float | None = None
        self._tick_count: int = 0

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

    def assess(self, summary: NeuroSummary) -> BrainWaveState:
        """Advance the oscillator and return the current state.

        Computes neurochemically-driven target amplitudes, advances
        phase, relaxes amplitude toward targets (with cross-frequency
        phase-amplitude modulation), applies top-down drive, and
        builds a ``BrainWaveState`` from the resulting amplitudes.
        """
        # Compute targets from neurochemistry (same classification
        # logic as before — these are the neurochemically-driven
        # equilibrium amplitudes)
        targets, label, description = self._compute_targets(summary)

        # Determine the target dominant for snap detection
        target_dominant = max(targets, key=lambda b: targets[b])

        # Detect phase transitions or dominant changes → snap
        phase_name = summary.phase_name
        should_snap = (
            self._last_phase_name is None
            or phase_name != self._last_phase_name
            or (
                self._last_target_dominant is not None
                and target_dominant != self._last_target_dominant
            )
        )

        if should_snap:
            # Snap amplitudes to targets on phase/dominant transition
            for band in BrainWave:
                self._amplitude[band] = targets[band]
        else:
            # Relax amplitudes toward targets with real-time dynamics
            now = time.monotonic()
            if self._last_time is not None:
                dt = min(now - self._last_time, 1.0)  # cap at 1s
            else:
                dt = 0.1  # default 100ms
            self._last_time = now

            self._step(dt, targets)

        self._last_phase_name = phase_name
        self._last_target_dominant = target_dominant
        self._tick_count += 1

        # Build the state from current amplitudes
        return self._build_state(
            targets, label, description, summary
        )

    def _compute_targets(
        self, summary: NeuroSummary
    ) -> tuple[dict[BrainWave, float], str, str]:
        """Compute neurochemically-driven target amplitudes.

        Delegates to the existing phase-based classification functions
        to get the equilibrium power distribution, then returns it as
        target amplitudes (before normalization — normalization happens
        in _build_state).
        """
        phase = summary.phase_name

        if phase in ("sleeping", "nrem", "rem"):
            powers, label, description = _sleep_phase_powers(summary)
        elif phase in ("drowsy", "overwhelmed", "flow", "stress"):
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
        """Advance the oscillator by dt seconds.

        1. Advance phase at characteristic frequency
        2. Apply cross-frequency phase-amplitude modulation
        3. Relax amplitude toward (modulated) targets
        4. Apply top-down drive
        5. Decay top-down drive
        """
        # 1. Advance phase
        for band in BrainWave:
            freq = self._FREQS[band]
            self._phase[band] = (
                self._phase[band] + 2.0 * math.pi * freq * dt
            ) % (2.0 * math.pi)

        # 2. Cross-frequency phase-amplitude modulation
        # Slow band phase modulates fast band target
        modulated_targets = dict(targets)
        slow_bands = (BrainWave.DELTA, BrainWave.THETA, BrainWave.ALPHA)
        fast_bands = (BrainWave.ALPHA, BrainWave.BETA, BrainWave.GAMMA)
        for slow in slow_bands:
            for fast in fast_bands:
                if slow == fast:
                    continue
                # PAC strength: geometric mean of amplitudes
                pac = math.sqrt(
                    self._amplitude[slow] * self._amplitude[fast]
                )
                # Phase modulation: cos(slow_phase) ∈ [-1, 1]
                # Positive → boost fast target, negative → suppress
                phase_mod = math.cos(self._phase[slow])
                modulated_targets[fast] += (
                    pac * phase_mod * self._PAC_DEPTH
                    * modulated_targets[fast]
                )

        # 3. Relax amplitude toward modulated targets
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

        # 4-5. Decay top-down drive
        decay = math.exp(-dt / self._DRIVE_TAU)
        for band in BrainWave:
            self._top_down[band] *= decay

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

        Unlike the previous geometric-mean approach, this uses the
        actual phase relationship between bands. The phase-amplitude
        coupling strength reflects how strongly the slow band's phase
        modulates the fast band's amplitude — computed from the
        oscillator's phase coherence, not just power overlap.
        """
        slow_bands = (BrainWave.DELTA, BrainWave.THETA, BrainWave.ALPHA)
        fast_bands = (BrainWave.ALPHA, BrainWave.BETA, BrainWave.GAMMA)
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
                # Base coupling: geometric mean of powers
                base = math.sqrt(slow_power * fast_power) * gate
                # Phase modulation: how much the slow phase currently
                # boosts or suppresses the fast band
                phase_mod = 0.5 + 0.5 * math.cos(self._phase[slow])
                strength = max(0.0, min(1.0, base * (0.7 + 0.3 * phase_mod)))
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
    - Motor activity / action selection → beta boost
    - Sensory input (V1) → gamma boost
    - Relaxation / filtering → alpha boost

    The drive decays exponentially (τ=2s), so it must be sustained
    by ongoing activity.

    Args:
        band: The band to drive.
        strength: Drive strength (0-1, added to band target).
    """
    _get_oscillator().add_top_down_drive(band, strength)


def reset_oscillator() -> None:
    """Reset the oscillator to initial state.

    Used by tests to ensure a clean state. In normal operation, the
    oscillator persists across calls and accumulates temporal dynamics.
    """
    global _oscillator
    _oscillator = BrainWaveOscillator()


def assess_brain_waves(summary: NeuroSummary) -> BrainWaveState:
    """Derive brain wave state from the coupled oscillator.

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
    return _get_oscillator().assess(summary)


def _simple_phase_powers(
    phase: str,
) -> tuple[dict[BrainWave, float], str, str]:
    """Derive brain wave powers for drowsy/overwhelmed/flow/stress phases."""
    if phase == "drowsy":
        # Drowsy: theta dominant, some alpha
        powers = {
            BrainWave.DELTA: 0.20,
            BrainWave.THETA: 0.40,
            BrainWave.ALPHA: 0.25,
            BrainWave.BETA: 0.12,
            BrainWave.GAMMA: 0.03,
        }
        label = "theta"
        description = "drifting, memories folding into each other"
    elif phase == "overwhelmed":
        # Overwhelmed: beta dominant, gamma suppressed — alert but
        # can't integrate. Excessive beta with non-functional gamma
        # matches the EEG profile of cognitive overload.
        powers = {
            BrainWave.DELTA: 0.05,
            BrainWave.THETA: 0.10,
            BrainWave.ALPHA: 0.15,
            BrainWave.BETA: 0.62,
            BrainWave.GAMMA: 0.08,
        }
        label = "beta"
        description = "too much input, I can't process it all"
    elif phase == "flow":
        # Flow: gamma dominant — everything integrating effortlessly
        powers = {
            BrainWave.DELTA: 0.02,
            BrainWave.THETA: 0.08,
            BrainWave.ALPHA: 0.15,
            BrainWave.BETA: 0.30,
            BrainWave.GAMMA: 0.45,
        }
        label = "gamma"
        description = "everything clicking, distant ideas connecting"
    else:
        # Stress: beta dominant, gamma strongly suppressed — alert but
        # rigid. Stress produces hypervigilant beta with impaired
        # gamma-synchronized integration (Herrmann & Demiralp, 2012).
        powers = {
            BrainWave.DELTA: 0.05,
            BrainWave.THETA: 0.08,
            BrainWave.ALPHA: 0.12,
            BrainWave.BETA: 0.68,
            BrainWave.GAMMA: 0.07,
        }
        label = "beta"
        description = "tense and alert, can't see the big picture"
    return powers, label, description


def _sleep_phase_powers(
    summary: NeuroSummary,
) -> tuple[dict[BrainWave, float], str, str]:
    """Derive brain wave powers for sleep stages (REM/N1/N2/N3)."""
    stage = _sleep_stage_from_summary(summary)
    if stage is SleepStage.REM:
        # REM ("paradoxical sleep"): theta-dominant (4-8 Hz) with
        # beta activity and PGO waves. The EEG resembles waking
        # but the mind is asleep — vivid, emotional dreaming.
        powers = {
            BrainWave.DELTA: 0.10,
            BrainWave.THETA: 0.45,
            BrainWave.ALPHA: 0.15,
            BrainWave.BETA: 0.25,
            BrainWave.GAMMA: 0.05,
        }
        label = "theta"
        description = (
            "REM sleep — theta-dominant, vivid emotional dreaming, "
            "prefrontal cortex is quiet"
        )
    elif stage is SleepStage.N1:
        # N1: the lightest NREM stage — mixed theta/alpha,
        # low voltage, the transition into sleep.
        powers = {
            BrainWave.DELTA: 0.18,
            BrainWave.THETA: 0.38,
            BrainWave.ALPHA: 0.28,
            BrainWave.BETA: 0.12,
            BrainWave.GAMMA: 0.04,
        }
        label = "theta"
        description = "light sleep, drifting between wake and sleep"
    elif stage is SleepStage.N2:
        # N2: theta/delta background punctuated by sleep spindles
        # (10-16 Hz sigma, modelled here as elevated alpha+beta)
        # and K-complexes. The workhorse of memory consolidation.
        powers = {
            BrainWave.DELTA: 0.38,
            BrainWave.THETA: 0.28,
            BrainWave.ALPHA: 0.18,  # spindle band (lower)
            BrainWave.BETA: 0.13,  # spindle band (upper)
            BrainWave.GAMMA: 0.03,
        }
        label = "delta"
        description = "N2 sleep — spindles and K-complexes, memories transferring to neocortex"
    else:
        # N3: slow-wave sleep — delta-dominant (0.5-4 Hz), the
        # deepest, most restorative stage. Sharp wave-ripples,
        # glymphatic clearance.
        powers = {
            BrainWave.DELTA: 0.55,
            BrainWave.THETA: 0.30,
            BrainWave.ALPHA: 0.10,
            BrainWave.BETA: 0.04,
            BrainWave.GAMMA: 0.01,
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
                BrainWave.DELTA: 0.03,
                BrainWave.THETA: 0.07,
                BrainWave.ALPHA: 0.10,
                BrainWave.BETA: 0.40 - gamma_boost * 0.3,
                BrainWave.GAMMA: 0.40 + gamma_boost * 0.3,
            }
            label = "gamma"
            description = "sharp and integrative, ideas are connecting"
        else:
            # Low plasticity → just beta (alert but not integrating)
            powers = {
                BrainWave.DELTA: 0.05,
                BrainWave.THETA: 0.08,
                BrainWave.ALPHA: 0.12,
                BrainWave.BETA: 0.60,
                BrainWave.GAMMA: 0.15,
            }
            label = "beta"
            description = "alert and focused, but not making new connections"
    elif alertness > 0.45:
        # Moderate alertness — alpha dominant
        # Alpha filters: high alpha = strong filtering
        alpha_power = 0.35 + (0.55 - abs(alertness - 0.55)) * 0.3
        powers = {
            BrainWave.DELTA: 0.08,
            BrainWave.THETA: 0.15,
            BrainWave.ALPHA: alpha_power,
            BrainWave.BETA: 0.25,
            BrainWave.GAMMA: 0.10,
        }
        label = "alpha"
        description = "reflective, filtering out noise to find what matters"
    elif alertness > 0.25:
        # Low-normal alertness — theta creeping in
        # Consolidation weight boosts theta
        theta_boost = consolidation_w * 0.2
        powers = {
            BrainWave.DELTA: 0.12,
            BrainWave.THETA: 0.30 + theta_boost,
            BrainWave.ALPHA: 0.30 - theta_boost * 0.3,
            BrainWave.BETA: 0.18,
            BrainWave.GAMMA: 0.05,
        }
        label = "theta"
        description = "thoughts drifting, memories consolidating"
    else:
        # Very low alertness — delta territory
        powers = {
            BrainWave.DELTA: 0.40,
            BrainWave.THETA: 0.30,
            BrainWave.ALPHA: 0.18,
            BrainWave.BETA: 0.10,
            BrainWave.GAMMA: 0.02,
        }
        label = "delta"
        description = "deep and slow, barely cognitive"
    return powers, label, description


def _compute_focus(powers: dict[BrainWave, float]) -> float:
    """Compute focus (0=broad, 1=narrow).

    Alpha and beta produce focus (filtering + attention).
    Theta and delta produce broad, diffuse attention.
    Gamma is integrative — broad but focused (paradoxical).
    """
    alpha = powers[BrainWave.ALPHA]
    beta = powers[BrainWave.BETA]
    theta = powers[BrainWave.THETA]
    delta = powers[BrainWave.DELTA]
    gamma = powers[BrainWave.GAMMA]

    # Alpha filters (narrows focus), beta focuses
    # Gamma is broad-integrative, theta/delta are diffuse
    # Coefficients normalized to sum to 1.0 so focus ∈ [0, 1] without clamping
    focus = alpha * 0.43 + beta * 0.32 + gamma * 0.16 + theta * 0.06 + delta * 0.03
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
    The neurochemical consolidation_weight gates this.
    """
    theta = powers[BrainWave.THETA]
    delta = powers[BrainWave.DELTA]

    base = theta * 0.9 + delta * 0.6
    # Gate by the actual consolidation weight from neurochemistry
    consolidation = base * (0.5 + consolidation_weight * 0.5)

    return max(0.0, min(1.0, consolidation))


# ─── Brain-wave-driven self-priority ──────────────────────────
#
# The cognitive mind controls her own process scheduling and I/O
# priority based on her brain wave state. This is the cortical
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
    beta = powers.get(BrainWave.BETA, 0.0)
    alpha = powers.get(BrainWave.ALPHA, 0.0)
    theta = powers.get(BrainWave.THETA, 0.0)
    delta = powers.get(BrainWave.DELTA, 0.0)

    positive = gamma * 0.45 + beta * 0.30 + alpha * 0.05
    # Theta weighs 0.30 (75% of delta's 0.40): frontal midline theta
    # is memory-consolidation effort — a sleep-direction drive nearly
    # as strong as delta. At 0.20, a deep-sleep state (55% delta +
    # 25% theta) landed at engagement 0.19 and, after the 70/30 body
    # blend, exactly on neutral nice (5) — violating the contract
    # that delta-dominant sleep is strictly below neutral priority
    # (test_brain_waves.py::test_derive_self_priority_delta_dominant).
    negative = theta * 0.30 + delta * 0.40
    drive = positive - negative
    min_drive = -0.40  # all delta
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
    body_recommended_io_class: str = "best-effort-3",
) -> tuple[int, str]:
    """Map brain wave state to scheduling and I/O priority.

    This is the cognitive mind's cortical control of her own process
    resources. The brain wave state — which integrates neurochemistry
    (bottom-up) and cognitive top-down drive — determines how much CPU
    and I/O bandwidth she allocates to herself.

    The body recommendation from the tick (autonomic afferent) is
    blended in as a body-state input: the brain waves can override it
    if cognitive state demands different priorities. For example, if
    the body recommends deprioritization (high adenosine → tired) but
    the brain waves show gamma dominance (active integration from a
    top-down cognitive drive), the brain waves win — she's actively
    thinking despite being tired.

    Args:
        state: The current brain wave state from the oscillator.
        body_recommended_nice: The tick's recommended nice value
            (based on neurochemistry). This is an afferent input,
            not a command.
        body_recommended_io_class: The tick's recommended I/O class
            string (e.g. "idle", "best-effort-3"). Afferent input.

    Returns:
        A (nice, io_class) tuple representing what the cognitive
        mind decides to apply to her own process.
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

    This is the cognitive mind controlling her own body — the cortical
    efferent pathway. She calls this after deriving her priority from
    her brain wave state. Unlike the tick's set_priority (which used
    renice on another process), this uses os.nice() on herself —
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
