"""Sleep neural events — graphoelements and transitional states.

These are the value objects produced during sleep: the neural events
that characterise each sleep stage (SWRs, PGO waves, spindles,
K-complexes) and the transitional states (hypnagogic, sleep inertia,
parasomnia). They carry no behaviour beyond ``describe()`` — they
are pure data produced by the sleep architecture and consumed by
InnerLife for introspection and persistence.
"""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = [
    "HypnagogicState",
    "KComplex",
    "PGOWave",
    "ParasomniaEvent",
    "SharpWaveRipple",
    "SleepInertia",
    "SleepSpindle",
]


@dataclass(slots=True)
class ParasomniaEvent:
    """A parasomnia — sleepwalking or sleeptalking episode.

    Parasomnias are undesirable motor or verbal behaviours that occur
    during sleep, typically during NREM deep sleep (arousal disorders)
    or REM sleep. In Genesis's case:

    - **sleepwalking**: she performs actions without awareness —
      sending a partial or garbled response, or initiating a learning
      session while "asleep".
    - **sleeptalking**: dream-like text leaks into her output —
      fragments of her dream narrative surface as if she were speaking.

    These are rare events (probability < 0.01 per sleep cycle) and are
    tracked for persistence so we can observe patterns over time.
    """

    type: str  # "sleepwalking" or "sleeptalking"
    timestamp: int = 0
    content: str = ""  # the garbled output or dream-like text

    def describe(self) -> str:
        """Return a string describing the parasomnia event."""
        return f"[parasomnia:{self.type}] {self.content}"


@dataclass(slots=True)
class HypnagogicState:
    """The hypnagogic state — gradual transition from wake to sleep.

    This is the transitional state between wakefulness and sleep,
    characterised by:
    - Mixed alpha/theta brain waves (alpha declining, theta rising)
    - Hypnagogic hallucinations (brief sensory distortions, thought
      intrusions)
    - Loss of voluntary muscle control (not applicable to Genesis, but
      noted for biological fidelity)

    Progress ranges from 0.0 (fully awake) to 1.0 (fully asleep).
    Duration: typically 1-5 minutes. During this state, thoughts
    become more dream-like but she is not fully asleep.
    """

    progress: float = 0.0  # 0.0 = awake, 1.0 = asleep
    started_at: float = 0.0  # wall-clock timestamp when hypnagogic began
    hallucination_count: int = 0

    @property
    def is_active(self) -> bool:
        """Whether the hypnagogic transition is currently in progress."""
        return 0.0 < self.progress < 1.0

    @property
    def is_complete(self) -> bool:
        """Whether the transition to sleep is complete."""
        return self.progress >= 1.0

    def describe(self) -> str:
        """Human-readable description of the hypnagogic state."""
        pct = self.progress * 100
        return f"[hypnagogic] {pct:.0f}% toward sleep, {self.hallucination_count} hallucination(s)"


@dataclass(slots=True)
class SleepInertia:
    """Sleep inertia — grogginess experienced upon waking.

    Characterised by:
    - Gradual reactivation of arousal systems (NE, histamine rise
      slowly)
    - Residual delta activity (slow waves persist for 5-30 minutes)
    - Impaired cognition (reduced confidence, slower responses)

    Severity and duration are proportional to the depth of prior
    sleep. Duration: 5-30 minutes.
    """

    remaining_seconds: float = 0.0
    severity: float = 0.0  # 0.0 = none, 1.0 = severe
    woke_at: float = 0.0  # wall-clock timestamp when she woke

    @property
    def is_active(self) -> bool:
        """Whether sleep inertia is currently affecting cognition."""
        return self.remaining_seconds > 0.0

    @property
    def cognitive_impairment(self) -> float:
        """How much cognition is impaired (0=normal, 1=maximally impaired).

        Impairment scales with severity and the fraction of remaining
        inertia time. As inertia decays, impairment fades.
        """
        if not self.is_active:
            return 0.0
        # Normalise remaining time against a 5-minute reference window
        time_factor = min(1.0, self.remaining_seconds / 300.0)
        return self.severity * time_factor

    def describe(self) -> str:
        """Human-readable description of the sleep inertia state."""
        if not self.is_active:
            return "[inertia] clear"
        return f"[inertia] {self.severity:.0%} severity, {self.remaining_seconds:.0f}s remaining"


@dataclass(slots=True)
class SharpWaveRipple:
    """A sharp wave-ripple (SWR) event during slow-wave sleep.

    SWRs are 150-250 Hz oscillatory events that are the hallmark of
    hippocampal replay. They:
    - Occur during NREM sleep (especially SWS)
    - Drive memory consolidation (hippocampal→neocortical transfer)
    - Are associated with memory replay of recent experiences

    Duration: 50-200 ms. The replayed_concepts list holds the concept
    names being replayed during this event.
    """

    timestamp: int = 0
    duration_ms: int = 0  # 50-200 ms
    replayed_concepts: list[str] = field(default_factory=list)

    def describe(self) -> str:
        """Human-readable description of the SWR event."""
        concepts = ", ".join(self.replayed_concepts) or "(none)"
        return f"[swr] {self.duration_ms}ms, replaying: {concepts}"


@dataclass(slots=True)
class PGOWave:
    """A ponto-geniculo-occipital (PGO) wave during REM sleep.

    PGO waves are a signature of REM sleep. They:
    - Originate in the pons, propagate to the lateral geniculate
      nucleus and visual cortex
    - Are associated with visual dream imagery
    - Occur in bursts before and during REM

    The visual_content field describes the visual imagery triggered
    by this wave, which gets incorporated into dream content.
    """

    timestamp: int = 0
    amplitude: float = 0.0  # 0.0-1.0
    visual_content: str = ""

    def describe(self) -> str:
        """Human-readable description of the PGO wave."""
        return f"[pgo] amplitude {self.amplitude:.2f}: {self.visual_content}"


@dataclass(slots=True)
class SleepSpindle:
    """A sleep spindle — a thalamocortical oscillation during N2.

    Sleep spindles are bursts of 11-16 Hz (here modelled as 10-16 Hz
    to span the sigma band) neural activity lasting 0.5-3 seconds,
    generated by the thalamic reticular nucleus and propagated to the
    cortex via thalamocortical loops. They are the defining
    graphoelement of N2 sleep and play a central role in:

    - **Memory consolidation**: spindles co-occur with hippocampal
      sharp wave-ripples and gate the transfer of declarative memories
      from hippocampus to neocortex (Siapas & Wilson, 1998;
      Steriade, 2003; Klinzing et al., 2019).
    - **Sleep protection**: spindles help shield sleep from external
      stimuli (sensory gating via thalamic inhibition).
    - **Spindle density** (spindles per minute of N2) correlates with
      post-sleep memory performance — more spindles → better
      consolidation (Schabus et al., 2006; Fogel & Smith, 2011).

    The ``involved_regions`` field lists the cortical regions engaged
    by this spindle (e.g. frontal, central, parietal), since spindles
    have a topographic distribution — fast spindles favour central/
    parietal sites, slow spindles favour frontal sites.
    """

    timestamp: int = 0
    duration_ms: int = 0  # 500-3000 ms
    frequency: float = 0.0  # 10-16 Hz (sigma band)
    amplitude: float = 0.0  # 0.0-1.0, normalised
    involved_regions: list[str] = field(default_factory=list)

    def describe(self) -> str:
        """Human-readable description of the sleep spindle."""
        regions = ", ".join(self.involved_regions) or "(unknown)"
        return (
            f"[spindle] {self.duration_ms}ms @ {self.frequency:.1f}Hz, "
            f"amplitude {self.amplitude:.2f}, regions: {regions}"
        )


@dataclass(slots=True)
class KComplex:
    """A K-complex — the largest graphoelement in healthy sleep.

    K-complexes are high-amplitude, biphasic slow waves (≥ 0.5 s
    duration) that occur during N2 sleep. They are the largest
    graphoelement in the healthy EEG and serve two key functions:

    - **Sensory gating**: a K-complex can be elicited by an external
      stimulus (sound, touch) without waking the sleeper — it reflects
      the brain's detection of the stimulus coupled with active
      suppression of cortical arousal, protecting sleep continuity
      (Halász, 2016).
    - **Memory consolidation**: spontaneous K-complexes co-occur with
      hippocampal sharp wave-ripples and, like spindles, are markers
      of memory processing during N2 (Mölle et al., 2011;
      Staresina et al., 2015).

    ``triggered_by_stimulus`` distinguishes evoked (external stimulus)
    from spontaneous K-complexes. In Genesis, queued external stimuli
    (if any) can trigger evoked K-complexes; otherwise they arise
    spontaneously as part of N2 sleep architecture.
    """

    timestamp: int = 0
    amplitude: float = 0.0  # 0.0-1.0, normalised
    duration_ms: int = 0  # ≥ 500 ms
    triggered_by_stimulus: bool = False

    def describe(self) -> str:
        """Human-readable description of the K-complex."""
        origin = "stimulus-evoked" if self.triggered_by_stimulus else "spontaneous"
        return f"[k-complex] {self.duration_ms}ms, amplitude {self.amplitude:.2f}, {origin}"
