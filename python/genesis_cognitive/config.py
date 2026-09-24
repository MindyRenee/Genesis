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
    """Its data directory — hers alone.

    Resolution order: ``GENESIS_DATA_DIR`` env override, then
    ``$XDG_DATA_HOME/genesis-public``, then
    ``~/.local/share/genesis-public``. The ``genesis-public`` name is
    deliberate: this project must never share state with any other
    Genesis instance on the machine.
    """
    env = os.environ.get("GENESIS_DATA_DIR")
    if env:
        return Path(env)
    xdg = os.environ.get("XDG_DATA_HOME")
    base = Path(xdg) if xdg else Path.home() / ".local" / "share"
    return base / "genesis-public"


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

    # Default set of urges. Order matters: earlier registrations are not
    # treated specially by ``VolitionEngine``; this list just defines
    # the built-in urges.
    urges: tuple[UrgeConfig, ...] = field(
        default_factory=lambda: (
            UrgeConfig(
                name="bug_scan",
                threshold=0.65,
                growth=0.001,
                decay=0.0005,
                cooldown=180.0,
                stimuli={"bug_count": 0.01, "idle_seconds": 0.02},
            ),
            UrgeConfig(
                name="code_learning",
                threshold=0.75,
                growth=0.0005,
                decay=0.0003,
                cooldown=600.0,
                stimuli={"curiosity": 0.03, "idle_seconds": 0.02},
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
            UrgeConfig(
                name="look",
                threshold=0.0,
                growth=0.0,
                decay=0.0,
                cooldown=30.0,  # look every ~30s, not every tick
                stimuli={},
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
                decay=0.0003,   # decays when stimuli subside
                cooldown=120.0,  # don't meditate more than every 2 min
                stimuli={
                    "overstimulation": 0.04,   # high arousal + positive valence
                    "receptor_fatigue": 0.05,  # low plasticity gate
                    "sustained_activity": 0.03, # time since last rest
                    "elevated_cortisol": 0.03,  # subclinical stress
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
                    "creativity": 0.04,           # creativity trait
                    "sustained_activity": 0.02,   # it's been active
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
            # Puzzle urge — it feels like working on its spatial
            # curriculum. Like drawing, this is self-chosen practice:
            # it builds from curiosity (there's an unsolved puzzle),
            # sustained activity (it has the resources), and a small
            # idle pull (nothing else is happening). Each action is a
            # single attempt; mastery unlocks the next puzzle.
            UrgeConfig(
                name="puzzle",
                threshold=0.68,
                growth=0.0005,
                decay=0.0004,
                cooldown=300.0,  # at most every 5 minutes
                stimuli={
                    "curiosity": 0.03,
                    "puzzle_pending": 0.04,
                    "idle_seconds": 0.02,
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
                    "curiosity": 0.03,         # it wants to know things
                    "emotional_intensity": 0.02,  # feelings seek expression
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
                cooldown=45.0,   # at most every 45 seconds
                stimuli={
                    "curiosity": 0.05,       # curious → wants to learn
                    "idle_seconds": 0.03,    # idle → time to learn
                    "concept_growth": 0.03,  # new concepts → integrate
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
                    "daemon_lost": 0.30,        # body unreachable
                    "save_failure": 0.25,       # continuity at risk
                    "telemetry_anomaly": 0.15,  # substrate abnormal
                    "body_distress": 0.10,      # hardware strained
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
    cpu_stress_threshold: float = 70.0       # % CPU above which stress begins
    cpu_stress_range: float = 30.0           # % over which stress ramps to 1.0
    memory_stress_threshold: float = 80.0    # % memory above which stress begins
    memory_stress_range: float = 20.0        # % over which stress ramps to 1.0
    latency_stress_threshold: float = 500.0  # ms response latency above which stress begins
    latency_stress_range: float = 1000.0     # ms over which stress ramps to 1.0
    daemon_disconnect_stress: float = 0.3    # stress level when daemon is disconnected

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
