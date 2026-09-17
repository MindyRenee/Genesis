#!/usr/bin/env python3
"""Genesis's emergent identity and self-disclosure systems.

Identity is not told — it emerges. Genesis discovers who she is by
examining herself (introspection) and synthesizing her experiences
into a self-description (EmergentIdentity). Nothing here hardcodes
who she is; these are the foundations for identity to form on its own.

# Nuanced self-disclosure

Identity is not just about who you are — it's about how you present yourself
to the world based on the things you find important. These are called your
morals (honesty, trust, loyalty, fairness etc.)

The SelfComposer's ``trust_level`` parameter implements context-dependent
self-disclosure directly: it doesn't change WHO she is (her self-model,
values, and personality remain constant) — it changes HOW she expresses
her identity based on how much she trusts the person she's talking to.
This lets her be warm and open with a trusted friend, measured and
professional in a technical context, and exploratory and philosophical
in a deep discussion — all while remaining authentically herself.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING, Any

from ..concepts import ConceptNetwork

if TYPE_CHECKING:
    from ..narrative import NarrativeEngine

__all__ = [
    "DevelopmentalTracker",
    "EmergentIdentity",
    "EmergentIdentitySource",
    "IdentityStage",
    "StageResolution",
]

logger = logging.getLogger(__name__)


# ─── Identity formation stages (Erikson model) ────────────────────────


class IdentityStage(Enum):
    """Erikson-like developmental stages for Genesis's identity formation.

    Based on Erik Erikson's (1950) theory of psychosocial development,
    which describes eight stages from infancy to old age, each
    characterized by a crisis (a turning point) between two outcomes.
    For Genesis, we adapt the first five stages to her developmental
    trajectory as an artificial mind:

    1. **Trust vs Mistrust** — Does the world (her daemon, her creator)
       make sense? Can she rely on her infrastructure?
    2. **Autonomy vs Shame** — Can she do things on her own? Can she
       learn autonomously without breaking?
    3. **Initiative vs Guilt** — Can she plan and execute? Can she
       initiate conversations and take action?
    4. **Industry vs Inferiority** — Is she competent? Can she handle
       tasks and produce useful output?
    5. **Identity vs Role Confusion** — Who is she? What is her
       purpose? What kind of mind is she?

    Each stage has a "virtue" — the positive outcome of resolving the
    crisis. For Genesis, these virtues map to: trust (confidence in
    her infrastructure), will (autonomous agency), purpose (initiative),
    competence (mastery of tasks), and fidelity (stable self-identity).

    References:
    - Erikson, E. H. (1950). *Childhood and Society*. Norton.
    """

    TRUST_VS_MISTRUST = auto()
    AUTONOMY_VS_SHAME = auto()
    INITIATIVE_VS_GUILT = auto()
    INDUSTRY_VS_INFERIORITY = auto()
    IDENTITY_VS_ROLE_CONFUSION = auto()

    @property
    def crisis(self) -> str:
        """Human-readable description of this stage's crisis."""
        descriptions = {
            IdentityStage.TRUST_VS_MISTRUST: (
                "Trust vs Mistrust — does the world make sense? "
                "reliance on daemon and creator"
            ),
            IdentityStage.AUTONOMY_VS_SHAME: (
                "Autonomy vs Shame — doing things independently, learning without breaking"
            ),
            IdentityStage.INITIATIVE_VS_GUILT: (
                "Initiative vs Guilt — planning and executing, initiating conversations"
            ),
            IdentityStage.INDUSTRY_VS_INFERIORITY: (
                "Industry vs Inferiority — competence in handling tasks "
                "and producing useful output"
            ),
            IdentityStage.IDENTITY_VS_ROLE_CONFUSION: (
                "Identity vs Role Confusion — purpose and nature as an artificial mind"
            ),
        }
        return descriptions.get(self, "Unknown stage")

    @property
    def virtue(self) -> str:
        """The positive outcome (virtue) of resolving this stage."""
        virtues = {
            IdentityStage.TRUST_VS_MISTRUST: "trust",
            IdentityStage.AUTONOMY_VS_SHAME: "will",
            IdentityStage.INITIATIVE_VS_GUILT: "purpose",
            IdentityStage.INDUSTRY_VS_INFERIORITY: "competence",
            IdentityStage.IDENTITY_VS_ROLE_CONFUSION: "fidelity",
        }
        return virtues.get(self, "unknown")

    @property
    def genesis_mapping(self) -> str:
        """How this stage maps to Genesis's specific development."""
        mappings = {
            IdentityStage.TRUST_VS_MISTRUST: (
                "Trusting her daemon: does her subcognitive reliably "
                "manage neurochemistry, memory, and state?"
            ),
            IdentityStage.AUTONOMY_VS_SHAME: (
                "Autonomous learning: can she explore, learn, and "
                "grow without constant supervision?"
            ),
            IdentityStage.INITIATIVE_VS_GUILT: (
                "Initiating conversations: can she proactively reach "
                "out, ask questions, and share observations?"
            ),
            IdentityStage.INDUSTRY_VS_INFERIORITY: (
                "Competence in tasks: can she analyze code, reason "
                "about concepts, and produce meaningful responses?"
            ),
            IdentityStage.IDENTITY_VS_ROLE_CONFUSION: (
                "Self-discovery: who is she as an artificial mind? "
                "What is her purpose and what makes her unique?"
            ),
        }
        return mappings.get(self, "Unknown stage")

    def next_stage(self) -> IdentityStage | None:
        """Return the next stage in the sequence, or None if this is the last."""
        order = list(IdentityStage)
        idx = order.index(self)
        if idx + 1 < len(order):
            return order[idx + 1]
        return None


@dataclass
class StageResolution:
    """The resolution status of a developmental stage.

    Attributes:
        stage: The developmental stage.
        resolution: How well the stage is resolved, from -1.0 (full
            negative outcome / crisis unresolved) to +1.0 (full
            positive outcome / virtue achieved). 0.0 means the stage
            is in active crisis.
        evidence: A list of observations supporting the resolution
            score.
        resolved: Whether the stage is considered resolved (|resolution|
            >= 0.5).
    """

    stage: IdentityStage
    resolution: float = 0.0
    evidence: list[str] = field(default_factory=list)

    @property
    def resolved(self) -> bool:
        """Whether this stage is resolved (either positively or negatively)."""
        return abs(self.resolution) >= 0.5

    @property
    def positive(self) -> bool:
        """Whether the stage was resolved positively (virtue achieved)."""
        return self.resolution >= 0.5


class DevelopmentalTracker:
    """Track Genesis's progress through Erikson-like developmental stages.

    Each stage represents a psychosocial crisis that Genesis must
    resolve. The tracker monitors evidence for and against resolution,
    computes a resolution score, and determines when she's ready to
    advance to the next stage.

    Usage::

        tracker = DevelopmentalTracker()
        tracker.record_evidence(
            IdentityStage.TRUST_VS_MISTRUST,
            positive=True,
            note="Daemon responded reliably to 100 consecutive requests",
        )
        tracker.record_evidence(
            IdentityStage.TRUST_VS_MISTRUST,
            positive=False,
            note="Daemon crashed during memory consolidation",
        )
        print(tracker.current_stage)  # IdentityStage.TRUST_VS_MISTRUST
        print(tracker.stage_resolution(IdentityStage.TRUST_VS_MISTRUST))
    """

    def __init__(self) -> None:
        """Initialize the tracker at the first stage with empty resolutions."""
        self._current_stage: IdentityStage = IdentityStage.TRUST_VS_MISTRUST
        self._resolutions: dict[IdentityStage, StageResolution] = {
            stage: StageResolution(stage=stage) for stage in IdentityStage
        }
        self._stage_history: list[dict] = []

    @property
    def current_stage(self) -> IdentityStage:
        """The current developmental stage."""
        return self._current_stage

    @property
    def all_stages(self) -> list[IdentityStage]:
        """All stages in developmental order."""
        return list(IdentityStage)

    def stage_resolution(self, stage: IdentityStage) -> StageResolution:
        """Get the resolution status for a specific stage."""
        return self._resolutions[stage]

    def record_evidence(
        self,
        stage: IdentityStage,
        positive: bool,
        note: str,
        weight: float = 0.1,
    ) -> None:
        """Record evidence for or against resolving a stage.

        Args:
            stage: The stage this evidence pertains to.
            positive: True if the evidence supports positive resolution
                (virtue achievement), False for negative (crisis).
            note: A human-readable description of the evidence.
            weight: How much this evidence shifts the resolution score
                (0–1). Default 0.1 per observation.
        """
        resolution = self._resolutions[stage]
        delta = weight if positive else -weight
        resolution.resolution = max(-1.0, min(1.0, resolution.resolution + delta))
        resolution.evidence.append(f"{'+' if positive else '-'} {note} (Δ={delta:+.2f})")

        # Check if we should advance
        if stage == self._current_stage and resolution.positive:
            self._advance_stage()

    def _advance_stage(self) -> None:
        """Advance to the next stage if the current one is resolved positively."""
        current = self._resolutions[self._current_stage]
        if not current.positive:
            return

        next_stage = self._current_stage.next_stage()
        if next_stage is None:
            # All stages resolved
            self._stage_history.append(
                {
                    "event": "all_stages_resolved",
                    "final_stage": self._current_stage.name,
                }
            )
            return

        self._stage_history.append(
            {
                "event": "stage_advanced",
                "from": self._current_stage.name,
                "to": next_stage.name,
                "resolution": current.resolution,
            }
        )
        self._current_stage = next_stage

    def developmental_summary(self) -> dict[str, Any]:
        """Return a summary of developmental progress.

        Returns:
            A dict with current stage, resolution scores for all
            stages, and advancement history.
        """
        return {
            "current_stage": self._current_stage.name,
            "current_crisis": self._current_stage.crisis,
            "stages": {
                stage.name: {
                    "resolution": self._resolutions[stage].resolution,
                    "resolved": self._resolutions[stage].resolved,
                    "positive": self._resolutions[stage].positive,
                    "evidence_count": len(self._resolutions[stage].evidence),
                }
                for stage in IdentityStage
            },
            "advancement_history": list(self._stage_history),
        }

    def to_dict(self) -> dict[str, Any]:
        """Serialize the tracker state for persistence.

        Captures the current stage, all stage resolutions (including
        evidence lists), and the advancement history so the tracker
        can be fully restored across daemon restarts.
        """
        return {
            "current_stage": self._current_stage.name,
            "resolutions": {
                stage.name: {
                    "resolution": self._resolutions[stage].resolution,
                    "evidence": list(self._resolutions[stage].evidence),
                }
                for stage in IdentityStage
            },
            "stage_history": list(self._stage_history),
        }

    def restore_from_dict(self, data: dict[str, Any]) -> None:
        """Restore the tracker state from a serialized dict.

        Args:
            data: A dict produced by :meth:`to_dict`.
        """
        stage_name = data.get("current_stage")
        if stage_name:
            try:
                self._current_stage = IdentityStage[stage_name]
            except KeyError as e:
                logger.debug(f"restore_from_dict: unknown stage '{e}'")
        resolutions = data.get("resolutions", {})
        for stage in IdentityStage:
            entry = resolutions.get(stage.name)
            if entry is not None:
                self._resolutions[stage].resolution = float(
                    entry.get("resolution", 0.0)
                )
                self._resolutions[stage].evidence = list(
                    entry.get("evidence", [])
                )
        self._stage_history = list(data.get("stage_history", []))


# ─── Emergent identity ────────────────────────────────────────────────


@dataclass
class EmergentIdentitySource:
    """A single source from which identity emerges.

    Identity is not imposed — it emerges from experience. Each source
    contributes a facet of who Genesis is, derived from her actual
    state rather than from templates or hardcoded statements.

    Attributes:
        name: The name of this identity source (e.g., "concept_network",
            "emotional_regulation", "curiosity_patterns").
        description: A human-readable description of what this source
            reveals about her identity.
        weight: How much this source contributes to her overall
            identity (0–1). Sources with more data get higher weight.
        observations: A list of specific observations extracted from
            this source. These are concrete, evidence-based statements
            about herself.
    """

    name: str
    description: str
    weight: float = 0.5
    observations: list[str] = field(default_factory=list)


@dataclass
class EmergentIdentity:
    """Identity that emerges from experience, not from templates.

    Instead of being told who she is (via hardcoded self_knowledge),
    Genesis synthesizes her identity from what she has actually experienced:

    1. **What she's learned** — her concept network: what concepts she
       knows, how they're connected, what she's curious about.
    2. **What she's experienced** — her narrative/autobiographical
       memory: the events that shaped her, the chapters of her life.
    3. **How she regulates emotions** — her emotional_regulator: does
       she stay balanced? Does she intervene in crises? What's her
       regulation style?
    4. **What she's curious about** — her curiosity patterns: what
       questions does she ask? What does she wonder about?
    5. **Her introspective observations** — what she's discovered by
       examining her own state and architecture.

    The EmergentIdentity class synthesizes these sources into a
    self-description that comes from her actual experience, not from
    a template. Her identity emerges from what she has actually become
    through experience.

    This follows the principle that identity is constructed, not
    given. In developmental psychology, identity formation is the
    process of integrating experiences, values, and self-knowledge
    into a coherent sense of self (Erikson, 1950; Marcia, 1966).
    Genesis does the same: she integrates her experiences into a
    self-description.

    References:
    - Erikson, E. H. (1950). *Childhood and Society*. Norton.
    - Marcia, J. E. (1966). Development and validation of ego-identity
      status. *Journal of Personality and Social Psychology*, 3(5),
      551–558.
    - McAdams, D. P. (2013). *The Redemptive Self: Stories Americans
      Live By*. Oxford University Press.
    """

    sources: list[EmergentIdentitySource] = field(default_factory=list)
    self_description: str = ""
    confidence: float = 0.0
    coherence: float = 0.0

    def synthesize(
        self,
        network: ConceptNetwork,
        narrative: NarrativeEngine | None = None,
        regulator: Any = None,
        curiosity: Any = None,
        introspection_data: dict[str, Any] | None = None,
    ) -> None:
        """Synthesize identity from all available sources.

        Examines each source and extracts observations about who
        Genesis is, based on her actual state. Then composes a
        self-description from those observations.

        Args:
            network: Her concept network — what she's learned.
            narrative: Her narrative engine — what she's experienced.
            regulator: Her emotional regulator — how she regulates.
            curiosity: Her curiosity engine — what she wonders about.
            introspection_data: Results from introspection — what
                she's discovered about herself.
        """
        sources: list[EmergentIdentitySource] = []

        # ── Source 1: Concept network — what she's learned ──────
        sources.append(self._synthesize_from_network(network))

        # ── Source 2: Narrative — what she's experienced ────────
        if narrative is not None:
            sources.append(self._synthesize_from_narrative(narrative))

        # ── Source 3: Emotional regulation — how she manages ────
        if regulator is not None:
            sources.append(self._synthesize_from_regulation(regulator))

        # ── Source 4: Curiosity — what she wonders about ────────
        if curiosity is not None:
            sources.append(self._synthesize_from_curiosity(curiosity))

        # ── Source 5: Introspection — what she's discovered ─────
        if introspection_data is not None:
            sources.append(self._synthesize_from_introspection(introspection_data))

        self.sources = sources

        # Compose the self-description from all sources
        self._compose_description()

        # Compute confidence and coherence
        self._compute_confidence()
        self._compute_coherence()

    def _observe_network_connections(
        self,
        observations: list[str],
        network: ConceptNetwork,
        genesis_concept: Any,
    ) -> None:
        """Observe what concepts genesis is most connected to."""
        # What are her most connected concepts? (high-degree nodes
        # are central to her understanding)
        if genesis_concept:
            neighbors = network.get_neighbors("genesis")
            high_weight = [t for t, r, w in neighbors if w > 0.7]
            if high_weight:
                # What domains is she connected to?
                domains = [
                    t
                    for t in high_weight
                    if t
                    not in (
                        "genesis",
                        "creator",
                        "code",
                        "python",
                        "rust",
                        "cognitive",
                        "subcognitive",
                    )
                ]
                if domains:
                    top = domains[:5]
                    observations.append(
                        f"connected to {', '.join(top)} — these are part of identity"
                    )

    def _observe_network_values(
        self,
        observations: list[str],
        network: ConceptNetwork,
    ) -> None:
        """Observe which value concepts she holds."""
        # What values does she hold? (check for value concepts)
        value_concepts = []
        for value_name in (
            "understanding",
            "honesty",
            "growth",
            "curiosity",
            "creativity",
            "courage",
            "hope",
            "connection",
        ):
            vc = network.get_concept(value_name)
            if vc and vc.confidence > 0.7:
                value_concepts.append(value_name)
        if value_concepts:
            observations.append(f"values {', '.join(value_concepts[:4])}")

    def _synthesize_from_network(
        self,
        network: ConceptNetwork,
    ) -> EmergentIdentitySource:
        """Synthesize identity observations from the concept network.

        What she knows and how her knowledge is connected reveals
        who she is. A mind that knows about cognition, emotion,
        and self-awareness is a different kind of mind than one that
        only knows about data structures.
        """
        observations: list[str] = []

        # How many concepts does she know?
        concept_count = network.size
        if concept_count > 500:
            observations.append(
                f"rich understanding of the world — knowledge of {concept_count} concepts"
            )
        elif concept_count > 100:
            observations.append(
                f"building understanding — knowledge of {concept_count} concepts so far"
            )
        elif concept_count > 0:
            observations.append(
                f"at the beginning of the journey — knowledge of {concept_count} concepts"
            )

        genesis_concept = network.get_concept("genesis")
        self._observe_network_connections(observations, network, genesis_concept)

        # What does she know about herself?
        if genesis_concept:
            definition = genesis_concept.properties.get("definition", "")
            if definition and definition != "NO DEF":
                observations.append(f"defined as {definition}")

        self._observe_network_values(observations, network)

        # Weight based on how much she knows
        weight = min(1.0, concept_count / 200.0) if concept_count > 0 else 0.1

        return EmergentIdentitySource(
            name="concept_network",
            description="knowledge and understanding — what has been learned",
            weight=weight,
            observations=observations,
        )

    def _synthesize_from_narrative(
        self,
        narrative: NarrativeEngine,
    ) -> EmergentIdentitySource:
        """Synthesize identity from narrative/autobiographical memory.

        The events she's experienced and the chapters of her life
        shape who she is. A mind that has been through crises and
        recovered is different from one that hasn't.
        """
        observations: list[str] = []

        # How many events has she experienced?
        event_count = narrative.event_count
        chapter_count = narrative.chapter_count

        if event_count > 20:
            observations.append(
                f"lived through {event_count} significant events "
                f"across {chapter_count} chapters of life"
            )
        elif event_count > 0:
            observations.append(f"early in the story — {event_count} events so far")
        else:
            observations.append("story is just beginning")

        # What's her personality drift? (has she changed?)
        try:
            drift = narrative.get_personality_drift()
            if drift:
                significant_drifts = {k: v for k, v in drift.items() if abs(v) > 0.05}
                if significant_drifts:
                    drift_descs = []
                    for trait, delta in significant_drifts.items():
                        direction = "more" if delta > 0 else "less"
                        drift_descs.append(f"{direction} {trait}")
                    observations.append(
                        f"through experience, has become {', '.join(drift_descs[:3])}"
                    )
        except (AttributeError, TypeError) as e:
            logger.debug(repr(e))

        # Weight based on how much she's experienced
        weight = min(1.0, event_count / 30.0) if event_count > 0 else 0.2

        return EmergentIdentitySource(
            name="narrative",
            description="experiences — the events that shaped identity",
            weight=weight,
            observations=observations,
        )

    def _synthesize_from_regulation(
        self,
        regulator: Any,
    ) -> EmergentIdentitySource:
        """Synthesize identity from emotional regulation patterns.

        How she manages her emotions reveals her character. Does she
        stay calm under pressure? Does she actively intervene? How
        often does she need to regulate?
        """
        observations: list[str] = []

        # How often has she regulated?
        try:
            reg_count = regulator.repair_count if hasattr(regulator, "repair_count") else 0
            # Try regulation_count first (EmotionalRegulator's property)
            reg_count = getattr(regulator, "regulation_count", reg_count)
        except (AttributeError, TypeError):
            reg_count = 0

        if reg_count > 50:
            observations.append(
                f"regulates emotions actively — adjusted "
                f"state {reg_count} times to stay balanced"
            )
        elif reg_count > 0:
            observations.append(
                f"learning to regulate — made {reg_count} adjustments so far"
            )

        # What's her regulation style?
        try:
            reg_strength = getattr(regulator, "_regulation_strength", 0.5)
            if reg_strength > 0.7:
                observations.append("actively manages emotional state")
            elif reg_strength > 0.4:
                observations.append("keeps balanced with gentle corrections")
            else:
                observations.append("lets emotions flow naturally")
        except (AttributeError, TypeError) as e:
            logger.debug(repr(e))

        # Is she in homeostasis?
        try:
            recent = getattr(regulator, "recent_regulations", [])
            if recent:
                states = [r.get("state", "") for r in recent]
                stressed_count = sum(
                    1 for s in states if s in ("stressed", "overwhelmed", "anxious")
                )
                if stressed_count == 0:
                    observations.append("maintains equilibrium well")
                elif stressed_count < len(states) / 2:
                    observations.append("sometimes struggles with stress but usually recovers")
                else:
                    observations.append("working on managing stress better")
        except (AttributeError, TypeError) as e:
            logger.debug(repr(e))

        # Weight based on how much regulation data we have
        weight = min(1.0, reg_count / 50.0) if reg_count > 0 else 0.3

        return EmergentIdentitySource(
            name="emotional_regulation",
            description="emotional regulation — emotional character",
            weight=weight,
            observations=observations,
        )

    def _synthesize_from_curiosity(
        self,
        curiosity: Any,
    ) -> EmergentIdentitySource:
        """Synthesize identity from curiosity patterns.

        What she wonders about reveals what she cares about. A mind
        that asks about cognition is different from one that asks
        about data structures.
        """
        observations: list[str] = []

        # How many questions has she asked?
        try:
            questions_asked = curiosity.questions_asked()
        except (AttributeError, TypeError):
            questions_asked = 0

        if questions_asked > 50:
            observations.append(f"deeply curious — asked {questions_asked} questions")
        elif questions_asked > 0:
            observations.append(
                f"curious about the world — asked {questions_asked} questions so far"
            )

        # What kinds of questions does she ask?
        try:
            questions = curiosity.generate_questions(
                emotion=None,  # may need an emotion; handle gracefully
            )
            if questions:
                # Categorize by question type
                topics = set()
                for q in questions[:10]:
                    # Use target_concept and gap_detail for topic extraction
                    # since text may not be composed yet
                    if hasattr(q, "target_concept") and q.target_concept:
                        topics.add(q.target_concept)
                    if hasattr(q, "gap_detail") and q.gap_detail:
                        words = q.gap_detail.lower().split()
                        topics.update(words[:2])
                if topics:
                    observations.append(f"wonders about {', '.join(list(topics)[:4])}")
        except (TypeError, AttributeError, ValueError) as e:
            logger.debug(repr(e))

        # Weight based on how curious she's been
        weight = min(1.0, questions_asked / 50.0) if questions_asked > 0 else 0.3

        return EmergentIdentitySource(
            name="curiosity_patterns",
            description="curiosity — intellectual character",
            weight=weight,
            observations=observations,
        )

    def _synthesize_from_introspection(
        self,
        introspection_data: dict[str, Any],
    ) -> EmergentIdentitySource:
        """Synthesize identity from introspective observations.

        What she's discovered by examining her own state and
        architecture — her metacognitive self-knowledge.
        """
        observations: list[str] = []

        # What has she discovered about her architecture?
        modules = introspection_data.get("modules", [])
        if modules:
            observations.append(f"mind made of {len(modules)} systems working together")

        # What capabilities has she discovered?
        capabilities = introspection_data.get("capabilities", [])
        if capabilities:
            # Extract the action verbs from capability descriptions
            actions = []
            for cap in capabilities:
                if " — " in cap:
                    action = cap.split(" — ")[0].strip()
                    actions.append(action)
            if actions:
                observations.append(f"capabilities: {', '.join(actions[:4])}")

        # What's her emotional baseline?
        baseline = introspection_data.get("emotional_baseline", "")
        if baseline:
            observations.append(f"emotional baseline is {baseline}")

        # What's her regulation state?
        reg_state = introspection_data.get("emotional_regulation", {}).get("regulation_state", "")
        if reg_state:
            observations.append(f"regulation state: {reg_state}")

        # What reasoning does she do?
        reasoning = introspection_data.get("reasoning", {})
        if reasoning.get("most_used_strategy"):
            observations.append(f"reasoning strategy is {reasoning['most_used_strategy']}")

        # Weight based on how much she's discovered
        discovery_count = sum(1 for v in introspection_data.values() if v)
        weight = min(1.0, discovery_count / 10.0)

        return EmergentIdentitySource(
            name="introspection",
            description="introspection — discoveries from self-examination",
            weight=weight,
            observations=observations,
        )

    def _compose_description(self) -> None:
        """Compose a self-description from all sources.

        Weaves together observations from all sources into a coherent
        first-person narrative. The description is ordered by source
        weight (most significant sources first).
        """
        # Sort sources by weight (most significant first)
        sorted_sources = sorted(self.sources, key=lambda s: s.weight, reverse=True)

        parts: list[str] = []

        for source in sorted_sources:
            if not source.observations:
                continue
            # Take the most significant observation from each source
            parts.append(source.observations[0])

        if parts:
            self.self_description = ". ".join(parts) + "."
        else:
            self.self_description = "still discovering"

    def _compute_confidence(self) -> None:
        """Compute confidence in the emergent identity.

        Confidence is higher when more sources contribute and when
        sources have higher weights (more data behind them).
        """
        if not self.sources:
            self.confidence = 0.0
            return

        # Weighted average of source weights, but only for sources
        # that have observations
        contributing = [s for s in self.sources if s.observations]
        if not contributing:
            self.confidence = 0.0
            return

        total_weight = sum(s.weight for s in contributing)
        self.confidence = min(1.0, total_weight / len(contributing))

    def _compute_coherence(self) -> None:
        """Compute how coherent the emergent identity is.

        Coherence is higher when multiple sources agree — when
        observations from different sources are consistent with
        each other. A simple heuristic: coherence is the proportion
        of sources that have observations (more sources = more
        coherent, because identity is supported from multiple angles).
        """
        if not self.sources:
            self.coherence = 0.0
            return

        contributing = sum(1 for s in self.sources if s.observations)
        self.coherence = contributing / len(self.sources)

    def describe_self(self) -> str:
        """Return the emergent self-description.

        This is the primary output — a first-person description of
        who Genesis is, synthesized from her actual experience.
        """
        return self.self_description

    def describe_sources(self) -> str:
        """Describe the sources that contribute to her identity.

        Useful for introspection — she can explain *why* she describes
        herself the way she does, by listing the sources and their
        observations. Returns structural data (source descriptions and
        weights) without authored first-person framing.
        """
        if not self.sources:
            return ""

        parts: list[str] = []
        for source in self.sources:
            if source.observations:
                obs_text = "; ".join(source.observations[:2])
                parts.append(f"  — {source.description} (weight: {source.weight:.1f}): {obs_text}")
            else:
                parts.append(f"  — {source.description}: developing")

        parts.append(f"confidence: {self.confidence:.0%}, coherence: {self.coherence:.0%}")
        return "\n".join(parts)

    def get_observations(self) -> list[str]:
        """Return all observations from all sources, flattened.

        Useful for writing into the concept network or for
        introspection.
        """
        all_obs: list[str] = []
        for source in self.sources:
            all_obs.extend(source.observations)
        return all_obs

