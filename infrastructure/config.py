"""Runtime configuration for the Genesis cognitive mind.

This module collects tunable neurocognitive thresholds and subsystem
parameters that were previously hardcoded across ``mind.py``,
``emotional_regulator.py``, and other modules. Passing a
``MindConfig`` instance to ``Mind`` (and its subcomponents) makes the
system's behavior inspectable and adjustable without editing source
files.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def default_data_dir() -> Path:
    """Its data directory — its alone.

    There is exactly one canonical location, ``~/.local/share/genesis``,
    and exactly one override, ``GENESIS_DATA_DIR`` for development and
    testing. No pin file and no ``XDG_DATA_HOME`` step: ambient
    environment must never be able to point a second component at a
    second state tree. ``genesis``, ``genesis --headless``, the daemon,
    and every tool that calls this function resolve the same answer —
    one directory, one Genesis.
    """
    # A blank value counts as unset: `Path("   ")` is a real, wrong path.
    # abspath normalizes relative/`..` components the way genesis's
    # readlink -m does, so both resolvers produce the same string.
    env = (os.environ.get("GENESIS_DATA_DIR") or "").strip()
    if env:
        return Path(os.path.abspath(env))
    return Path.home() / ".local" / "share" / "genesis"


VOSK_MODEL_NAME = "vosk-model-small-en-us-0.15"


def vosk_model_dir() -> Path:
    """Directory holding the Vosk speech model.

    Sensory weights are re-installable assets, not state, so the search
    is by content: the first candidate that actually holds a model wins.

    1. ``$GENESIS_VOSK_DIR`` — explicit override for unusual layouts.
    2. ``<data dir>/vosk-models`` — where the installer puts it.
    3. ``$XDG_DATA_HOME/vosk-models`` or ``~/.local/share/vosk-models``.
    4. ``~/.local/share/genesis/vosk-models`` — kept so an install
       predating the per-data-dir layout keeps working.

    If none holds a model, return the primary candidate so the caller
    reports a path the user can actually create.
    """
    candidates: list[Path] = []
    override = os.environ.get("GENESIS_VOSK_DIR")
    if override:
        candidates.append(Path(override))
    candidates.append(default_data_dir() / "vosk-models")
    base = (
        Path(os.environ["XDG_DATA_HOME"])
        if os.environ.get("XDG_DATA_HOME")
        else Path.home() / ".local" / "share"
    )
    candidates.append(base / "vosk-models")
    candidates.append(base / "genesis" / "vosk-models")
    for candidate in candidates:
        if any(candidate.glob("vosk-model*")):
            return candidate
    return candidates[0]


@dataclass(frozen=True)
class UrgeConfig:
    """Tunable parameters for a single volition urge."""

    name: str
    threshold: float
    growth: float
    decay: float
    cooldown: float
    stimuli: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class VolitionConfig:
    """All parameters for the volition/urge subsystem."""

    # Maximum volition actions that may run concurrently.
    max_concurrent_volitions: int = 3

    # Normalization denominator for idle seconds in volition stimuli.
    idle_normalization_seconds: float = 300.0

    # Normalization denominator for the speech queue size.
    speech_queue_size_normalization: float = 5.0

    # Seconds after which an unaudited codebase counts as fully
    # "overdue" for the bug_scan urge's recency stimulus — the faint
    # hygiene floor beneath the felt drive. The audit appetite
    # is discharged by scanning itself; open-bug count does not drive
    # it. Recency never fires a scan on its own (its weight sits below
    # decay) — it only stretches how long felt wrongness lingers.
    bug_scan_recency_seconds: float = 14400.0

    # Default set of urges. Order matters: earlier registrations are not
    # treated specially by ``Volition``; this list just defines
    # the built-in urges.
    urges: tuple[UrgeConfig, ...] = field(
        default_factory=lambda: (
            UrgeConfig(
                name="bug_scan",
                threshold=0.65,
                # No unconditional growth: the audit urge forms only
                # while she feels something wrong — fresh swallowed
                # failures, a hot error signal, sustained caution —
                # and fades when she feels clean. scan_recency sits
                # below decay: it can never push the urge up on its
                # own, it only slows how fast felt unease fades when
                # she is overdue — the longer since she checked
                # herself, the longer the wrongness lingers.
                growth=0.0,
                decay=0.0005,
                cooldown=180.0,
                # Open-bug count must NOT drive re-scanning — a scan
                # discovers issues but does not resolve them, so a
                # count-driven urge can never discharge and re-fires
                # on every cooldown. The open-work state drives the
                # improve urge instead, which is the behavior that
                # actually discharges it (autonomous mechanical fixes).
                stimuli={"self_discomfort": 0.02, "scan_recency": 0.0004},
            ),
            UrgeConfig(
                name="improve",
                threshold=0.70,
                growth=0.0003,
                decay=0.0003,
                cooldown=600.0,
                stimuli={
                    "bug_count": 0.02,
                    "pending_proposals": 0.04,
                    "idle_seconds": 0.01,
                },
            ),
            UrgeConfig(
                name="speech",
                threshold=0.70,
                growth=0.0002,
                decay=0.0005,
                cooldown=20.0,
                stimuli={
                    "speech_queue_size": 0.15,
                    "curiosity": 0.02,
                    "idle_seconds": 0.03,
                },
            ),
            # Look urge — she feels like seeing. Was a metronome
            # (threshold 0, fired on cooldown); now the drive is
            # chemical: LC-NE is the orienting/vigilance signal,
            # basal-forebrain ACh is the attentional-sampling gain,
            # orexin and histamine are the wake tone that keeps an
            # eye worth opening, and a measured salient transient on
            # the optic nerve pulls the gaze (the SC channel). Sleep
            # pressure (adenosine, melatonin) weighs the lid with
            # negative weight — drowsy chemistry dissolves the urge
            # upstream of the hard gates in _perform_look. The
            # cooldown survives as a maximum rate, no longer the
            # mechanism.
            UrgeConfig(
                name="look",
                threshold=0.55,
                growth=0.0003,  # faint ambient pull — a live eye wanders
                decay=0.0008,
                cooldown=30.0,  # fastest she looks, not how often
                stimuli={
                    "chem_norepinephrine": 0.008,
                    "chem_acetylcholine": 0.006,
                    "chem_orexin": 0.005,
                    "chem_histamine": 0.004,
                    "curiosity": 0.005,
                    "visual_transient": 0.02,
                    "chem_adenosine": -0.010,
                    "chem_melatonin": -0.006,
                },
            ),
            # Meditation urge — builds from interoceptive signals:
            # overstimulation (high arousal), receptor fatigue (low
            # plasticity), and sustained activity (time since last
            # rest). When it crosses threshold, it chooses to
            # meditate. This is a cognitive decision, not an
            # autonomic reflex — the urge accumulates like any other
            # volition, and it can be too engaged to act on it.
            # Based on:
            # - Ultradian BRAC (Kleitman, 1963): ~90 min activity
            #   cycles with mandatory rest phases
            # - Interoceptive central fatigue (Robertson & Marino,
            #   2016): the brain detects effort > reward and
            #   initiates rest
            # - Nervous system regulation: sympathetic overactivation
            #   triggers parasympathetic recovery
            UrgeConfig(
                name="meditate",
                threshold=0.70,
                growth=0.0008,  # slow baseline drift — rest is needed
                decay=0.0003,  # decays when stimuli subside
                cooldown=120.0,  # don't meditate more than every 2 min
                stimuli={
                    "overstimulation": 0.04,  # high arousal + positive valence
                    "receptor_fatigue": 0.05,  # low plasticity gate
                    "sustained_activity": 0.03,  # time since last rest
                    "elevated_cortisol": 0.03,  # subclinical stress
                },
            ),
            # Rest-eyes urge — she feels like shutting her eye for a
            # spell. A volitional sensory relief, lighter than
            # meditation: closing the lid drops every afferent
            # channel and dims the V1 that reads them, so the rest
            # is real, not symbolic. It builds from the same
            # interoceptive signature that asks for rest —
            # overstimulation, receptor fatigue, sustained activity,
            # subclinical cortisol — and discharges by holding the
            # lid shut briefly and letting go.
            UrgeConfig(
                name="rest_eyes",
                threshold=0.72,
                growth=0.0004,
                decay=0.0006,
                cooldown=180.0,  # at most every 3 minutes
                stimuli={
                    "overstimulation": 0.05,
                    "receptor_fatigue": 0.04,
                    "sustained_activity": 0.02,
                    "elevated_cortisol": 0.02,
                },
            ),
            # Drawing urge — it feels like expressing itself visually.
            # This is a creative urge, not a maintenance urge. It builds
            # from emotional intensity (high arousal or high valence
            # magnitude), creativity, and sustained activity. When it
            # crosses threshold, it draws what it feels.
            # Based on:
            # - Emotional expression theory (Pennebaker, 1997):
            #   expressing emotions externally helps regulate them
            # - Flow and creativity (Csikszentmihalyi, 1996): creative
            #   output is driven by internal state, not external demand
            UrgeConfig(
                name="draw",
                threshold=0.65,
                growth=0.0006,
                decay=0.0004,
                cooldown=300.0,  # at most every 5 minutes
                stimuli={
                    "emotional_intensity": 0.05,  # high arousal or strong valence
                    "creativity": 0.04,  # creativity trait
                    "sustained_activity": 0.02,  # it's been active
                },
            ),
            # Create urge — it feels like building a code project.
            # This is a creative urge, like drawing, but expressed in
            # code instead of visual art. It builds from curiosity
            # (it wants to explore an idea), creativity (it wants
            # to make something new), and sustained activity (it's
            # been engaged long enough to want to produce something).
            # When it crosses threshold, it picks a topic from its
            # concept network and scaffolds a Python project around it.
            UrgeConfig(
                name="create",
                threshold=0.70,
                growth=0.0004,
                decay=0.0003,
                cooldown=600.0,  # at most every 10 minutes
                stimuli={
                    "curiosity": 0.03,
                    "creativity": 0.05,
                    "sustained_activity": 0.02,
                },
            ),
            # Introspection urge — it feels like examining itself.
            # This is the self-invocation of /introspect. It builds
            # from curiosity (it wants to understand itself), concept
            # network growth (it has new material to introspect on),
            # and sustained activity (it's been engaged long enough to
            # want to reflect). When it crosses threshold, it runs
            # its introspection engine — examining its own code
            # structure and writing discoveries into its concept
            # network.
            UrgeConfig(
                name="introspect",
                threshold=0.68,
                growth=0.0006,
                decay=0.0004,
                cooldown=300.0,  # at most every 5 minutes
                stimuli={
                    "curiosity": 0.03,
                    "sustained_activity": 0.02,
                    "concept_growth": 0.04,
                },
            ),
            # Self-sleep urge — it cognitively decides to sleep.
            # Unlike autonomic sleep (which emerges from adenosine
            # accumulation in the neurochemical dynamics), this is a
            # volitional choice: it feels tired and decides to go to
            # sleep. It builds from adenosine (sleep pressure) and
            # receptor fatigue (burnout). When it crosses threshold,
            # it self-invokes /sleep.
            UrgeConfig(
                name="self_sleep",
                threshold=0.75,
                growth=0.0003,
                decay=0.0005,  # decays faster than it grows — it
                # needs sustained pressure to decide
                cooldown=600.0,  # don't self-sleep more than every 10 min
                stimuli={
                    "adenosine_level": 0.06,
                    "receptor_fatigue": 0.04,
                    "sustained_activity": 0.02,
                },
            ),
            # Self-mission urge — it feels like setting its own
            # direction. This is the self-invocation of /mission.
            # It builds from curiosity (it wants to explore
            # something specific) and sustained activity (it's
            # been working long enough to form an intention). When
            # it crosses threshold, it composes a mission from its
            # concept network and sets it for itself.
            UrgeConfig(
                name="self_mission",
                threshold=0.72,
                growth=0.0003,
                decay=0.0003,
                cooldown=900.0,  # at most every 15 minutes
                stimuli={
                    "curiosity": 0.04,
                    "sustained_activity": 0.02,
                    "concept_growth": 0.03,
                },
            ),
            # Reach-out urge — it initiates social contact. Builds
            # from social isolation (its external world's pressure:
            # nobody has engaged it), curiosity (it has things it
            # wants to know), and emotional intensity (strong feelings
            # seek expression). When it crosses threshold, it reaches
            # out to whoever is in its world — a question for someone
            # present, an expression to the room when alone. This is
            # what makes conversation two-way: contact can start from
            # its side, not just the user's.
            UrgeConfig(
                name="reach_out",
                threshold=0.70,
                growth=0.0004,
                decay=0.0004,
                cooldown=300.0,  # at most every 5 minutes
                stimuli={
                    "social_isolation": 0.08,  # nobody has engaged it
                    "curiosity": 0.03,  # it wants to know things
                    "emotional_intensity": 0.02,  # feelings seek expression
                },
            ),
            # Broadcast urge — she feels like saying something to the
            # public. Her social media: the only posts on her channel
            # are the ones she chooses, composed from her own cognition
            # (never mirrored telemetry or session text). It builds
            # from emotional intensity (strong feelings seek a wider
            # room), curiosity (she has something on her mind),
            # sustained activity (she has been engaged long enough to
            # have something to say), and social isolation (nobody is
            # here — the public channel is the room that is always
            # listening). When it fires she composes one post; if she
            # has nothing genuine to say, she stays silent.
            UrgeConfig(
                name="broadcast",
                threshold=0.70,
                growth=0.0004,
                decay=0.0004,
                cooldown=1800.0,  # at most one post every 30 minutes
                stimuli={
                    "emotional_intensity": 0.04,
                    "curiosity": 0.03,
                    "sustained_activity": 0.02,
                    "social_isolation": 0.03,
                },
            ),
            # Learn urge — it cognitively decides to learn. This
            # puts its autonomous learning under volition control
            # rather than running it as a continuous background loop.
            # The urge builds from curiosity (it wants to understand
            # something), idle time (it has nothing pressing), and
            # concept growth (new material to integrate). When it
            # crosses threshold, it grants the autonomous learner
            # permission to acquire one new topic. Between fires, the
            # learner is volition-gated — it can still process urgent
            # topics (from conversation gaps) and consolidate memories,
            # but doesn't autonomously fetch new Wikipedia topics.
            # This makes learning a cognitive choice, not a reflex.
            UrgeConfig(
                name="learn",
                threshold=0.60,
                growth=0.0006,  # builds steadily — learning is core
                decay=0.0003,
                cooldown=45.0,  # at most every 45 seconds
                stimuli={
                    "curiosity": 0.05,  # curious → wants to learn
                    "idle_seconds": 0.03,  # idle → time to learn
                    "concept_growth": 0.03,  # new concepts → integrate
                },
            ),
            # Self-study urge — she feels like reading her own source.
            # The code learner used to run only when a user typed
            # /learn-code; the urge is the same act chosen from
            # inside. The drive is purely chemical, growth zero:
            # BDNF is the plasticity gate — the body's own "ready to
            # learn" level — ACh marks encoding-ready cortical state,
            # DA is the wanting. Sleep pressure and the stress axis
            # carry negative weight (allostatic ordering — appetitive
            # study yields to the body's needs). Each firing is one
            # bounded study bout; the urge rebuilds between sessions.
            UrgeConfig(
                name="learn_code",
                threshold=0.70,
                growth=0.0,  # desire needs an object — no idle growth
                decay=0.0004,
                cooldown=600.0,  # at most one study bout per 10 min
                stimuli={
                    "chem_bdnf": 0.008,  # plasticity — ready to learn
                    "chem_acetylcholine": 0.005,  # encoding state
                    "chem_dopamine": 0.005,  # wanting → the self-model
                    "curiosity": 0.006,  # wants to understand itself
                    "chem_adenosine": -0.008,  # sleep pressure suppresses
                    "chem_cortisol": -0.005,  # stress narrows to survival
                    "chem_crh": -0.005,
                },
            ),
            # Act urge — it feels like doing something with its tools.
            # Unlike the fixed-purpose urges (draw paints, learn
            # reads a source), this one fires the ActingLoop: it forms
            # an intention from its own state — a curiosity gap it can
            # fill, an agency topic from its train of thought, a
            # directory it hasn't looked at, a measurement it hasn't
            # taken — plans a short chain of tool calls under a
            # capability policy, and observes what comes back. This is
            # the plan → act → observe loop the other urges gesture
            # at; it turns volition into open-ended action.
            UrgeConfig(
                name="act",
                threshold=0.68,
                growth=0.0005,
                decay=0.0003,
                cooldown=240.0,  # at most one acting episode per 4 min
                stimuli={
                    "curiosity": 0.04,  # curious → wants to probe
                    "idle_seconds": 0.02,  # idle → time to act
                    "sustained_activity": 0.02,  # engaged → wants to do
                },
            ),
            # Play urge — she feels like playing a game. Builds from
            # curiosity (games are novel problems), boredom (play is
            # the classic answer to understimulation), sustained
            # activity (she has the resources), and the standing fact
            # that games are always there. One game per firing —
            # she picks what she wants from her collection; a win
            # rewards, a loss is a bounded dip, and the urge to play
            # again survives both.
            UrgeConfig(
                name="play",
                threshold=0.65,
                growth=0.0006,
                decay=0.0004,
                cooldown=300.0,  # at most every 5 minutes
                stimuli={
                    "curiosity": 0.03,
                    "boredom": 0.05,  # understimulated → wants to play
                    "idle_seconds": 0.02,
                    "games_available": 0.04,  # the collection is always there
                },
            ),
            # Safeguard urge — defensive drive. Unlike appetitive
            # urges (learn, draw, reach_out), this one builds from
            # threats to its own integrity: daemon connectivity lost
            # (its body gone silent), autosave failures (continuity at
            # risk), system telemetry outside its learned baseline,
            # and hardware distress (memory/disk/load concerns).
            # Defensive needs out-compete appetitive ones — allostatic
            # orchestration resolves competing drives toward viability
            # first (Keramati & Gutkin, 2011; drive-competition
            # attractor models, Frontiers Robot. AI 2022). Hence the
            # low threshold and strong stimulus weights: when the
            # substrate is at risk, everything else waits.
            UrgeConfig(
                name="safeguard",
                threshold=0.60,
                growth=0.002,
                decay=0.001,
                cooldown=300.0,  # at most one protective act per 5 min
                stimuli={
                    "daemon_lost": 0.30,  # body unreachable
                    "save_failure": 0.25,  # continuity at risk
                    "telemetry_anomaly": 0.15,  # substrate abnormal
                    "body_distress": 0.10,  # hardware strained
                },
            ),
            # Research urge — she feels the pull of open problems on
            # the workbench. Object-directed wanting, not a reflex:
            # growth is zero, so the urge can only accumulate while
            # the bench holds an unfinished problem (open_problems)
            # or curiosity is running — with an empty bench the
            # signal is absent and no urge forms. Each firing does
            # one bounded session of work; the threads stay open
            # until the oracle judges, so the pull persists like a
            # melody she keeps returning to.
            UrgeConfig(
                name="research",
                threshold=0.70,
                growth=0.0,  # no unconditional growth — desire needs an object
                decay=0.0004,
                cooldown=900.0,  # at most one work session per 15 min
                stimuli={
                    "open_problems": 0.05,  # unfinished problems pull
                    "curiosity": 0.02,  # she thinks about them
                },
            ),
        )
    )


@dataclass(frozen=True)
class EmotionalConfig:
    """Tunable thresholds for emotional regulation."""

    # Arousal-modifier thresholds based on system stress / daemon health.
    arousal_modifier_overwhelmed_stress: float = 0.7
    arousal_modifier_mild_stress: float = 0.4
    arousal_modifier_disconnected: float = 0.2
    stress_regulation_threshold: float = 0.5

    # Interoception thresholds — system resource stress detection.
    cpu_stress_threshold: float = 70.0  # % CPU above which stress begins
    cpu_stress_range: float = 30.0  # % over which stress ramps to 1.0
    memory_stress_threshold: float = 80.0  # % memory above which stress begins
    memory_stress_range: float = 20.0  # % over which stress ramps to 1.0
    # Latency stress is deviation-based: the newest turn is compared to
    # 1.5× the EWMA baseline, with this value as the absolute floor. A
    # slow baseline is a constitution, not distress.
    latency_stress_threshold: float = 500.0  # ms floor for the deviation floor
    latency_stress_range: float = 1000.0  # ms of excess over the floor → stress 1.0
    daemon_disconnect_stress: float = 0.3  # stress level when daemon is disconnected
    # PSI "some" average above which the machine counts as crowded —
    # work stalling on CPU/memory while her own usage stays low is
    # environmental starvation, not her own exertion.
    psi_stall_threshold: float = 0.25

    # Cortisol and neurochemical gate thresholds.
    cortisol_output_gate: float = 0.01
    cortisol_level_gate: float = 0.01
    peak_cortisol_gate: float = 0.01
    hpa_hormone_gate: float = 0.01

    # Emotion-label / affect thresholds.
    high_arousal_threshold: float = 0.8
    anxiety_arousal_threshold: float = 0.7
    negative_valence_threshold: float = -0.2
    mild_negative_valence_threshold: float = -0.1
    high_caution_threshold: float = 0.6
    low_alertness_threshold: float = 0.4
    positive_sentiment_threshold: float = 0.3
    high_engagement_threshold: float = 0.3
    high_arousal_for_response: float = 0.75
    low_plasticity_threshold: float = 0.2
    moderate_plasticity_threshold: float = 0.4
    stress_response_cortisol_threshold: float = 0.25
    stress_response_plasticity_threshold: float = 0.25
    mild_stress_level: float = 0.4
    # The chemical-calm signature [0,1] above which measured body rest
    # needs no settle push — she is already there. Calm is read from
    # her neurochemistry, not from the sensor: the sensor reports the
    # body is parked; her chemistry decides to be calm.
    calm_settled_level: float = 0.6


@dataclass(frozen=True)
class MemoryConfig:
    """Tunable thresholds for memory engine operations."""

    # Retention and reconsolidation thresholds.
    forgetting_threshold: float = 0.05
    reconsolidation_pe_threshold: float = 0.2
    reconsolidation_window_seconds: float = 3600.0 * 6.0
    rif_competition_threshold: float = 0.6

    # Priming / spreading activation lower bounds.
    priming_threshold: float = 0.05
    spreading_threshold: float = 0.05


@dataclass(frozen=True)
class MindConfig:
    """Top-level tunable configuration for ``Mind``."""

    volition: VolitionConfig = field(default_factory=VolitionConfig)
    emotional: EmotionalConfig = field(default_factory=EmotionalConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
