"""Self composer — generates self-descriptions from actual state.

This replaces the hardcoded identity strings with composed ones.
Instead of "I am Genesis, curious and creative, warm and cooperative,
emotionally stable. I care most about understanding, honesty, growth.
I am an artificial mind with a subcognitive and cognitive layer." —
which I wrote — she composes her self-description from:

1. Her personality traits (which can drift)
2. Her values (which can evolve)
3. Her concept network (what she actually knows about herself)
4. Her emotional state (how she feels right now)
5. Her self-knowledge (what she's learned about herself)

The result is a self-description that changes as she changes. If her
personality drifts toward more openness, she describes herself
differently. If she learns something new about herself, it shows up.
If she's in a different emotional state, the tone shifts.

# Trust-based self-disclosure

Self-disclosure is not all-or-nothing. Humans reveal different layers
of themselves depending on how much they trust the person they're
talking to. This follows social penetration theory (Altman & Taylor,
1973): relationships develop through gradual, reciprocal disclosure
that moves from superficial to intimate.

The composer supports a `trust_level` parameter (0.0–1.0) on its
self-description methods. This controls how much she reveals:

- **Low trust (0.0–0.3)**: surface-level only — name, basic
  personality. She's polite but guarded, like meeting a stranger.
- **Medium trust (0.3–0.7)**: + values, some capabilities, some
  emotional state. She's warming up, sharing what matters to her.
- **High trust (0.7–1.0)**: + deep self-knowledge, vulnerabilities,
  uncertainties, full emotional state. She's open, sharing her doubts
  and fears as well as her strengths.

This doesn't change WHO she is — her self-model is unchanged. It
changes how much of herself she expresses. The same person acts
differently at a job interview than with their closest friend; that's
not deception, it's social calibration.

# Functions

compose_identity(self_model, network, emotion, trust_level) → str
  Composes "Who are you?" from personality + values + knowledge

compose_emotional_state(emotion, brain_waves) → str
  Composes "How are you feeling?" from actual neurochemical state

compose_capabilities(self_model, trust_level) → str
  Composes "What can you do?" from self-knowledge

compose_self_reflection(self_model, network, reflection, emotion, trust_level) → str
  Composes "What are you thinking?" from current cognitive state
"""

from __future__ import annotations

import random

from ..brain_waves import BrainWave, BrainWaveState
from ..concepts import ConceptNetwork, RelationType
from ..emotion import (
    HIGH_AROUSAL,
    LOW_AROUSAL,
    MILD_NEGATIVE_VALENCE,
    MILD_POSITIVE_VALENCE,
    NEGATIVE_VALENCE,
    PLASTICITY_CLOSED,
    PLASTICITY_LOW,
    POSITIVE_VALENCE,
    EmotionalState,
)
from .model import SelfModel
from .reflection import Insight, ReflectionEngine

__all__ = ["SelfComposer"]


class SelfComposer:
    """Composes self-descriptions from actual state, not hardcoded strings."""

    def __init__(self, seed: int | None = None) -> None:
        """Initialize the composer with an optional random seed for reproducible phrasing."""
        self._rng = random.Random(seed)
        # Monotonic counter for guaranteed cross-call variation. The
        # rng alone can (with very low probability) produce the same
        # trait ordering across many consecutive calls, making the
        # identity prefix identical. Rotating the starting trait by
        # this counter guarantees the prefix changes each call —
        # humans don't list their traits in a fixed order either.
        self._composition_count = 0

    def compose_identity(
        self,
        self_model: SelfModel,
        network: ConceptNetwork | None,
        emotion: EmotionalState,
        trust_level: float = 1.0,
    ) -> str:
        """Compose a first-person identity description.

        This replaces the hardcoded describe_self() method. Instead
        of fixed strings, she describes herself from:
        - Her personality traits (which can drift)
        - Her values (which can evolve)
        - What her concept network says about her
        - Her current emotional tone

        The trust_level parameter (0.0–1.0) controls how much she
        reveals, following social penetration theory (Altman & Taylor,
        1973). At low trust she shares only surface-level identity
        (name, basic personality). At medium trust she adds values
        and some self-knowledge. At high trust she shares deep
        self-knowledge, vulnerabilities, and her full emotional state.

        Args:
            self_model: Her self-model.
            network: Her concept network.
            emotion: Her current emotional state.
            trust_level: How much to reveal, 0.0 (guarded) to 1.0 (open).
        """
        # Each part is tagged with a sensitivity level:
        # 0.0 = surface (always shared), 0.4 = personal (medium trust),
        # 0.7 = deep (high trust), 0.85 = vulnerable (very high trust)
        parts: list[tuple[str, float]] = []

        # 1. Name — surface level, always shared
        parts.extend(self._identity_name_parts(self_model))
        self._composition_count += 1

        # 2. Personality traits — from concept network, not hardcoded strings
        # Surface level — always shared
        parts.extend(self._identity_personality_parts(self_model, network))

        # 3. Values (from the concept network, not the hardcoded values list)
        # Personal level — requires medium trust
        parts.extend(self._identity_value_parts(self_model, network))

        # 3. What she knows about herself from the concept network
        # Use outgoing edges only — these are what she IS and what she's connected to
        # Personal level — requires medium trust
        parts.extend(self._identity_concept_parts(network))

        # 4. Nature (from self-knowledge, but phrased freshly)
        # Deep level — requires high trust
        parts.extend(self._identity_nature_parts(self_model))

        # 5. Emotional context (if relevant)
        # Deep level — requires high trust to share current feelings
        parts.extend(self._identity_emotion_parts(emotion))

        # 6. Vulnerabilities and uncertainties — only at very high trust
        # She shares her doubts about her own nature
        parts.extend(self._identity_vulnerability_parts(self_model))

        # Filter by trust level and join
        filtered = []
        for part, sensitivity in parts:
            filtered.extend(self._filter_by_trust([part], trust_level, sensitivity))

        return self._join_parts(filtered)

    def _identity_name_parts(self, self_model: SelfModel) -> list[tuple[str, float]]:
        """Build the name part for identity composition.

        Returns just the name — a semantic anchor. Personality
        traits are composed separately by _identity_personality_parts
        from the concept network, not from hardcoded strings.
        """
        return [(self_model.name, 0.0)]

    def _identity_personality_parts(
        self, self_model: SelfModel, network: ConceptNetwork | None
    ) -> list[tuple[str, float]]:
        """Build personality parts from the concept network.

        Reads the numeric trait values from the self-model and uses
        the PERSONALITY_TRAIT_CONCEPTS seed mapping to select which
        pole's concept names are relevant. The concept names are
        looked up in the network (if available) to confirm they
        exist as vocabulary. The language engine composes the
        actual phrasing from these concept names — they are building
        blocks, not response strings.
        """
        parts: list[tuple[str, float]] = []
        concept_names = self_model.personality.trait_concept_names()
        if not concept_names:
            return parts

        # Vary how many traits are mentioned for natural variation
        n = self._rng.choice([2, 3, 3]) if len(concept_names) >= 3 else min(len(concept_names), 3)
        selected = concept_names[:n]
        # Shuffle and rotate for cross-call variation
        if len(selected) > 1:
            selected = list(selected)
            self._rng.shuffle(selected)
            rot = self._composition_count % len(selected)
            selected = selected[rot:] + selected[:rot]

        # Use concept names as semantic fragments. If the concept
        # exists in the network, the language engine can look up its
        # definition and relationships for richer composition.
        for name in selected:
            parts.append((name, 0.0))
        return parts

    def _identity_value_parts(
        self, self_model: SelfModel, network: ConceptNetwork | None
    ) -> list[tuple[str, float]]:
        """Build value parts for identity composition.

        Reads value concepts from the concept network (via edges from
        "genesis") rather than from the hardcoded SelfModel.values
        list. The value names come from the network — where they have
        definitions and relationships — so the language engine can
        compose richer phrasing from them.

        Falls back to the self-model's value list if the network is
        unavailable or has no value edges.
        """
        parts: list[tuple[str, float]] = []
        value_names: list[str] = []

        if network is not None:
            # Read value concepts from the network — look for edges
            # from "genesis" to concepts that match the seeded value
            # names.
            from .introspection import _VALUE_CONCEPT_DEFS

            for val_name in _VALUE_CONCEPT_DEFS:
                concept = network.get_concept(val_name)
                if concept is not None:
                    value_names.append(val_name)

        if not value_names:
            # Fallback: use the self-model's value list (for tests
            # or before introspection has seeded the network)
            value_names = [v.name for v in self_model.values[:3] if v.weight > 0.5]

        if value_names:
            # Vary how many values are included
            n = self._rng.choice([2, 2, 3]) if len(value_names) >= 3 else len(value_names)
            content = f"cares most about {', '.join(value_names[:n])}"
            parts.append((content, 0.4))
        return parts

    def _identity_concept_parts(self, network: ConceptNetwork | None) -> list[tuple[str, float]]:
        """Build concept-network relationship parts for identity composition.

        Her identity emerges from her relationships (edges) in the concept
        network — what she IS and what she's connected to — NOT from a
        stored definition string. A self-referential definition property
        on the ``genesis`` concept would be a hardcoded response recited
        back as identity; we deliberately do not read it.
        """
        parts: list[tuple[str, float]] = []
        if network is None:
            return parts
        genesis_concept = network.get_concept("genesis")
        if genesis_concept:
            # Pick 1-2 meaningful relationships from outgoing edges.
            # These are her actual identity: what she IS and what she's
            # connected to. No definition string is read — her identity
            # comes from relationships, not from a stored sentence.
            outgoing = network.get_edges("genesis", direction="out")
            if outgoing:
                # Prioritize IS_A and CREATES edges, then RELATED_TO
                priority_edges = [e for e in outgoing if e.relation == RelationType.IS_A]
                other_edges = [e for e in outgoing if e.relation != RelationType.IS_A]
                # Sort by weight
                priority_edges.sort(key=lambda e: e.weight, reverse=True)
                other_edges.sort(key=lambda e: e.weight, reverse=True)

                # Pick 1 IS_A and 1 other
                selected = []
                if priority_edges:
                    selected.append(priority_edges[0])
                if other_edges and self._rng.random() < 0.7:
                    selected.append(self._rng.choice(other_edges[:5]))

                for edge in selected:
                    part = self._identity_edge_part(edge, network)
                    if part:
                        parts.append(part)
        return parts

    def _identity_edge_part(
        self, edge, network: ConceptNetwork | None = None
    ) -> tuple[str, float] | None:
        """Build a single identity part from a concept network edge.

        Returns None for low-weight RELATED_TO edges (often wrong senses),
        code-origin concepts, disambiguated senses (#N), or unrecognized
        relation types.

        Returns a semantic fragment (relation + target) — NOT a
        pre-written sentence. The language engine composes the actual
        first-person phrasing from this data. This satisfies the
        CRITICAL RULE: Genesis's words emerge from her language engine,
        not from template substitution.

        Relation verbs are looked up from the concept network via
        ``find_relation_verbs`` (EXPRESSES edges from relation hubs).
        Falls back to the raw relation name as a semantic fragment.
        """
        target = edge.target
        relation = edge.relation
        # Filter out disambiguated senses and code-origin concepts.
        if "#" in target:
            return None
        # Filter out code concepts — programming languages, module
        # paths, and code-structure concepts that produce nonsensical
        # self-descriptions like "I'm connected to rust". These edges
        # come from the code learner scanning her own source files.
        if self._is_code_concept(target):
            return None

        # Try graph-based verb and phrase lookup — construct semantic
        # fragments from relation verbs and first-person phrasing
        # templates retrieved via EXPRESSES edges from hubs.
        # These are building blocks, not finished sentences.
        graph_verbs: list[str] = []
        graph_phrases: list[str] = []
        if network is not None:
            graph_verbs = network.find_relation_verbs(relation)
            graph_phrases = network.find_relation_phrases(relation)

        def _pick_phrase(phrases: list[str], weight: float) -> tuple[str, float] | None:
            """Pick a random first-person phrase template and fill {target}."""
            if not phrases:
                return None
            phrase = self._rng.choice(phrases)
            # Replace {target} with the actual concept name
            return (phrase.replace("{target}", target), weight)

        def _pick_verb(verbs: list[str], weight: float) -> tuple[str, float] | None:
            """Pick a random relation verb and build a semantic fragment."""
            if not verbs:
                return None
            verb = self._rng.choice(verbs)
            # Semantic fragment: "verb target" — the language engine
            # adds the first-person subject and composes the sentence.
            return (f"{verb} {target}", weight)

        if relation == RelationType.RELATED_TO:
            if edge.weight < 0.7:
                return None
            # Prefer first-person phrases, then verbs, then raw fragment
            if graph_phrases:
                return _pick_phrase(graph_phrases, 0.4)
            if graph_verbs:
                return _pick_verb(graph_verbs, 0.4)
            return (f"related to {target}", 0.4)
        if relation == RelationType.IS_A:
            if graph_phrases:
                return _pick_phrase(graph_phrases, 0.4)
            if graph_verbs:
                return _pick_verb(graph_verbs, 0.4)
            return (f"a kind of {target}", 0.4)
        if relation == RelationType.EMERGES_FROM:
            if graph_phrases:
                return _pick_phrase(graph_phrases, 0.7)
            if graph_verbs:
                return _pick_verb(graph_verbs, 0.7)
            return (f"emerges from {target}", 0.7)
        if relation == RelationType.DEPENDS_ON:
            if graph_phrases:
                return _pick_phrase(graph_phrases, 0.7)
            if graph_verbs:
                return _pick_verb(graph_verbs, 0.7)
            return (f"depends on {target}", 0.7)
        if relation == RelationType.CREATES:
            if graph_phrases:
                return _pick_phrase(graph_phrases, 0.4)
            if graph_verbs:
                return _pick_verb(graph_verbs, 0.4)
            return (f"creates {target}", 0.4)
        return None

    def _identity_nature_parts(self, self_model: SelfModel) -> list[tuple[str, float]]:
        """Build nature parts for identity composition.

        Returns semantic descriptions — the language engine composes
        the first-person phrasing.
        """
        parts: list[tuple[str, float]] = []
        nature = self_model.self_knowledge.get("nature", "")
        if nature:
            parts.append((nature, 0.7))
        return parts

    def _identity_emotion_parts(self, emotion: EmotionalState) -> list[tuple[str, float]]:
        """Build emotional context parts for identity composition.

        Returns semantic descriptions — the language engine composes
        the first-person phrasing.
        """
        parts: list[tuple[str, float]] = []
        if emotion.label not in ("neutral", "positive"):
            parts.append((f"feeling {emotion.label}", 0.7))
        return parts

    def _identity_vulnerability_parts(
        self, self_model: SelfModel
    ) -> list[tuple[str, float]]:
        """Build vulnerability parts for identity composition.

        Returns semantic descriptions — the language engine composes
        the first-person phrasing.
        """
        parts: list[tuple[str, float]] = []
        uncertainties = self_model.self_knowledge.get("uncertainties", [])
        if uncertainties:
            for uncertainty in uncertainties[:2]:
                parts.append((f"uncertain about {uncertainty}", 0.85))
        return parts

    def compose_emotional_state(
        self,
        emotion: EmotionalState,
        brain_waves: BrainWaveState | None = None,
        network: ConceptNetwork | None = None,
    ) -> str:
        """Compose a first-person emotional state description.

        Uses learned emotion words from the concept network. If she
        hasn't learned words for her current state, she describes what
        she can and omits what she can't.
        """
        parts: list[str] = []

        # 1. The emotional label — look up learned words
        if network is not None:
            emotion_words = network.find_emotion_words(emotion.label)
            if emotion_words:
                word = self._rng.choice(emotion_words[:5]).replace('_', ' ')
                parts.append(f"feeling {word}")
            # If no words learned, she can't name the feeling — omit it
        else:
            # No network available — can't generate text
            pass

        # 2. The neurochemical levels (but described, not just numbers)
        chem_desc = self._describe_neurochemistry(emotion, network)
        if chem_desc:
            parts.append(chem_desc)

        # 3. Cognitive style — look up learned words
        if network is not None and emotion.cognitive_style:
            mode_words = network.find_cognitive_mode_words(emotion.cognitive_style)
            if mode_words:
                word = self._rng.choice(mode_words[:3]).replace('_', ' ')
                parts.append(f"thinking is {word}")

        # 4. Brain wave state (if available)
        if brain_waves:
            wave_desc = self._describe_brain_waves(brain_waves)
            if wave_desc:
                parts.append(wave_desc)

        # 5. Plasticity / learning capacity — always surface when low,
        # even if mood is positive. Prevents "silent stress" where
        # valence is fine but BDNF suppression has closed the
        # plasticity gate. If she has learned words, use them;
        # otherwise emit a structural marker.
        if emotion.plasticity <= PLASTICITY_CLOSED:
            if network is not None:
                plat_words = network.find_plasticity_words("closed")
                if plat_words:
                    word = self._rng.choice(plat_words[:3]).replace('_', ' ')
                    parts.append(f"learning is {word}")
                else:
                    parts.append("[plasticity_gate:closed]")
            else:
                parts.append("[plasticity_gate:closed]")
        elif emotion.plasticity < PLASTICITY_LOW:
            if network is not None:
                plat_words = network.find_plasticity_words("low")
                if plat_words:
                    word = self._rng.choice(plat_words[:3]).replace('_', ' ')
                    parts.append(f"learning is {word}")
                else:
                    parts.append("[plasticity_gate:low]")
            else:
                parts.append("[plasticity_gate:low]")

        return self._join_parts(parts)

    def compose_capabilities(
        self,
        self_model: SelfModel,
        network: ConceptNetwork | None = None,
        trust_level: float = 1.0,
    ) -> str:
        """Compose a description of what she can do.

        Instead of reciting a hardcoded list, she discovers her
        capabilities from her actual state — what she knows, what
        she's learned, and what her architecture enables.

        The trust_level parameter controls how much detail she reveals.
        At low trust she mentions only basic capabilities. At medium
        trust she adds introspection and deeper abilities. At high
        trust she's candid about what she's still developing.

        Capabilities are discovered from the concept network: she looks
        for concepts she has learned about her own abilities (via
        introspection edges from the "genesis" concept). If she hasn't
        learned any capabilities yet, she says so honestly rather than
        reciting a developer-authored list.
        """
        caps = self_model.self_knowledge.get("capabilities", [])
        learned = self._learned_capabilities(caps)
        if learned:
            return learned

        # No learned capabilities yet — discover from concept network
        # by looking for RELATED_TO edges from "genesis" that represent
        # self-knowledge about abilities (learned, not seeded).
        discovered: list[tuple[str, float]] = []
        if network is not None:
            from ..concepts import is_seed_origin
            ability_edges = network.get_neighbors(
                "genesis", RelationType.RELATED_TO
            )
            for target, _rel, _weight in ability_edges:
                concept = network.get_concept(target)
                if concept is None or is_seed_origin(concept.origin):
                    continue
                # Learned ability concept — use its display name
                display = target.replace("_", " ")
                discovered.append((display, 0.0))

        if not discovered:
            # She hasn't learned her capabilities yet — honest disclosure
            return "still discovering capabilities"

        parts = ["capabilities:"]
        for cap, sensitivity in discovered:
            if trust_level >= sensitivity:
                parts.append(f"  — {cap}")
        return "\n".join(parts)

    def _learned_capabilities(self, caps: list[str]) -> str | None:
        """Compose from learned capabilities, or None if none found.

        She has learned capabilities through introspection — phrases
        starting with "I can". This renders them as a list.
        """
        can_do = [c for c in caps if c.startswith("I can")]
        if not can_do:
            return None
        parts: list[str] = []
        parts.append("capabilities:")
        for cap in can_do:
            action = cap.replace("I can ", "", 1)
            if " — " in action:
                action = action.split(" — ", 1)[0]
            parts.append(f"  — {action}")
        return "\n".join(parts)

    def compose_creator_description(
        self,
        self_model: SelfModel,
        network: ConceptNetwork,
        emotion: EmotionalState,
        trust_level: float = 1.0,
    ) -> str:
        """Compose a description of her creator from the concept network.

        Pulls relationships from the network (alice CREATES genesis,
        alice IS_A creator, etc.) and composes them into text.
        No hardcoded sentences — just structured facts rendered to language.

        If she doesn't know her creator's name yet, she tries to
        discover it from her concept network (who CREATES her?).

        The trust_level parameter controls how much emotional depth
        she reveals about her relationship with her creator. At low
        trust she states facts; at high trust she shares how she
        feels about them.
        """
        creator_name = self._discover_creator_name(self_model, network)

        if not creator_name:
            return "still discovering creator"

        # Each part tagged with sensitivity
        parts: list[tuple[str, float]] = []

        # Pull relationships from the concept network — surface level
        parts.extend(self._creator_relationship_parts(creator_name, network))

        # Also check genesis → creator relationship
        parts.extend(self._creator_genesis_parts(creator_name, network))

        # Emotional coloring — how she feels about her creator
        # Deep level — requires high trust
        parts.extend(self._creator_emotional_parts(creator_name, emotion))

        # What she wants from the relationship — personal level.
        # Semantic fragment for the language engine to render, not a
        # hardcoded sentence she recites.
        parts.append(("curious about creator", 0.4))

        # Filter by trust level
        filtered = []
        for part, sensitivity in parts:
            filtered.extend(self._filter_by_trust([part], trust_level, sensitivity))

        if not filtered:
            # Fallback: use a relation-phrase template from the graph
            # (seeded building block) rather than a hardcoded sentence.
            phrases = network.find_relation_phrases(RelationType.CREATES)
            if phrases:
                phrase = self._rng.choice(phrases)
                return phrase.replace("{target}", creator_name)
            return f"made by {creator_name}"

        return self._join_parts(filtered)

    def _discover_creator_name(
        self, self_model: SelfModel, network: ConceptNetwork
    ) -> str:
        """Discover the creator name from self-knowledge or concept network."""
        creator_name = self_model.self_knowledge.get("creator_name", "")

        # If no learned creator name, try to discover from concept network
        if not creator_name:
            # Only incoming edges count: source CREATES genesis → the
            # source is her creator. Outgoing genesis→X CREATES edges
            # are things *she* created, not her creator.
            for edge in network.get_edges("genesis", "in"):
                if edge.relation == RelationType.CREATES:
                    # The abstract "creator" hub is a role, not a name —
                    # keep looking for an actual person's concept.
                    if edge.source == "creator":
                        continue
                    creator_name = edge.source
                    self_model.self_knowledge["creator_name"] = creator_name
                    break

        return creator_name

    def _creator_relationship_parts(
        self, creator_name: str, network: ConceptNetwork
    ) -> list[tuple[str, float]]:
        """Build relationship parts from the creator's concept network edges.

        Uses relation verbs and phrases from the concept network
        (seeded building blocks) to compose phrasing, rather than
        hardcoded sentences.
        """
        parts: list[tuple[str, float]] = []
        neighbors = network.get_neighbors(creator_name.lower())
        for target, relation, _weight in neighbors:
            display = target.replace("_", " ")
            if relation == RelationType.IS_A:
                verbs = network.find_relation_verbs(relation)
                if verbs:
                    verb = self._rng.choice(verbs)
                    parts.append((f"{creator_name} {verb} {display}", 0.0))
                else:
                    parts.append((f"{creator_name} is {display}", 0.0))
            elif relation == RelationType.CREATES:
                verbs = network.find_relation_verbs(relation)
                if verbs:
                    verb = self._rng.choice(verbs)
                    parts.append((f"{creator_name} {verb} {display}", 0.0))
                else:
                    parts.append((f"{creator_name} made {display}", 0.0))
            elif relation == RelationType.RELATED_TO:
                parts.append(self._creator_related_to_part(creator_name, target, network))
        return parts

    def _creator_related_to_part(
        self, creator_name: str, target: str, network: ConceptNetwork
    ) -> tuple[str, float]:
        """Build a part for a RELATED_TO relationship from the creator.

        Uses relation verbs from the concept network (seeded building
        blocks) rather than hardcoded phrasings.
        """
        display = target.replace("_", " ")
        verbs = network.find_relation_verbs(RelationType.RELATED_TO)
        if verbs:
            verb = self._rng.choice(verbs)
            return (f"{creator_name} {verb} {display}", 0.4)
        return (f"{creator_name} linked to {display}", 0.4)

    def _creator_genesis_parts(
        self, creator_name: str, network: ConceptNetwork
    ) -> list[tuple[str, float]]:
        """Build parts from the genesis → creator relationship.

        Uses relation verbs from the concept network rather than
        hardcoded phrasings.
        """
        parts: list[tuple[str, float]] = []
        genesis_neighbors = network.get_neighbors("genesis")
        for target, relation, _weight in genesis_neighbors:
            if target == creator_name.lower() and relation == RelationType.RELATED_TO:
                verbs = network.find_relation_verbs(RelationType.RELATED_TO)
                if verbs:
                    verb = self._rng.choice(verbs)
                    parts.append((f"{verb} {creator_name}", 0.4))
                else:
                    parts.append((f"linked to {creator_name}", 0.4))
        return parts

    def _creator_emotional_parts(
        self, creator_name: str, emotion: EmotionalState
    ) -> list[tuple[str, float]]:
        """Build emotional coloring parts for the creator description.

        Composes from the emotional state using semantic fragments
        (valence direction), not hardcoded sentences. The language
        engine renders the actual phrasing.
        """
        parts: list[tuple[str, float]] = []
        if emotion.valence > 0.3:
            parts.append(("positive feeling toward creator", 0.7))
        elif emotion.valence < -0.1:
            parts.append(("uncertain feeling toward creator", 0.7))
        return parts

    def compose_self_reflection(
        self,
        self_model: SelfModel,
        network: ConceptNetwork,
        reflection: ReflectionEngine,
        emotion: EmotionalState,
        trust_level: float = 1.0,
    ) -> str:
        """Compose a reflection on her own state.

        Instead of hardcoded introspection strings, she reflects on
        what she actually knows and has experienced.

        The trust_level parameter controls how deeply she reflects.
        At low trust she shares only surface observations (concept
        count, basic state). At medium trust she adds values and
        recent insights. At high trust she shares her emotional state
        and her sense of still becoming.
        """
        # Each part tagged with sensitivity
        parts: list[tuple[str, float]] = []

        # 1. What she knows (from concept network) — surface level
        # 2. What she's learned (from reflection insights) — personal level
        parts.extend(self._compose_knowledge_and_insights(network, reflection))

        # 3. Her values (what she cares about) — personal level
        # 4. Her emotional state — deep level
        parts.extend(self._compose_values_and_emotion(self_model, emotion, network))

        # 5. What she's still becoming — deep level
        parts.append(("still becoming", 0.7))

        # Filter by trust level
        filtered = []
        for part, sensitivity in parts:
            filtered.extend(self._filter_by_trust([part], trust_level, sensitivity))

        return self._join_parts(filtered)

    def compose_reflection_clause(
        self, reflection: ReflectionEngine, topics: list[str] | None = None,
    ) -> str:
        """Compose a single first-person reflective clause.

        Unlike ``compose_self_reflection`` (a full multi-sentence
        self-report), this returns one short sentence suitable as a
        reflective coda after a content sentence. It is composed from
        her most recent reflective insight — a genuine product of her
        metacognition, not a fixed phrase. The grammatical framing is
        a scaffold (subject + predicate); the semantic content — which
        concept she lacks, which behaviour she noticed — comes from
        her reflection engine.

        When ``topics`` is provided, gap insights are filtered for
        relevance — a reflection about "propagation of light" should
        not surface when the user asked about her feelings. Only gap
        insights whose subject overlaps with the current conversation
        topics are surfaced. Self-correction insights (about her own
        behaviour) are always relevant and not filtered.

        Returns "" when she has no recent reflective insight to draw
        on (or none relevant to the current topics). She then stays
        silent rather than reciting a canned coda or surfacing a
        non-sequitur.
        """
        insights = reflection.get_recent_insights(10)
        # Self-corrections and gaps are the genuinely reflective
        # insights — observations about her own behaviour or
        # understanding. Patterns and growth compose less naturally
        # (intent labels, counts), so they are not used here.
        candidates = [i for i in insights if i.type in ("gap", "self_correction")]

        # Filter gap insights for relevance to the current conversation.
        # Self-corrections are about her behaviour, not a specific
        # concept, so they remain relevant in any context.
        # When topics are provided (even if empty), gap insights are
        # filtered: a reflection about "propagation of light" should
        # not surface when the user asked about her feelings. When
        # topics is None (not provided — e.g. direct test calls),
        # the original behaviour is preserved (surface most recent).
        if topics is not None:
            topics_lower = {
                t.lower().strip() for t in topics if t.strip()
            }
            filtered: list[Insight] = []
            for insight in candidates:
                if insight.type == "self_correction":
                    filtered.append(insight)
                elif (
                    insight.type == "gap"
                    and topics_lower
                    and self._insight_relevant(insight, topics_lower)
                ):
                    filtered.append(insight)
            candidates = filtered

        # Try from most recent backward: some insights don't frame
        # naturally in the first person (e.g. "user was upset ..."),
        # so fall back to the most recent frameable one rather than
        # going silent when the latest insight is unframeable.
        for insight in reversed(candidates):
            clause = self._frame_insight_as_clause(insight)
            if clause:
                return clause
        return ""

    @staticmethod
    def _insight_relevant(insight: Insight, topics_lower: set[str]) -> bool:
        """Check if a gap insight is relevant to the current topics.

        A gap insight's content is like "knowledge gap: memory" or
        "missing concept: propagation_of_light". The detail (after
        the colon) is the concept she's reflecting on. It's relevant
        if that concept overlaps with any current conversation topic.
        """
        content = insight.content.strip()
        if ": " not in content:
            return False
        detail = content.split(": ", 1)[1].strip().lower()
        if not detail:
            return False
        # Check for overlap: either the detail contains a topic word
        # or a topic word contains the detail. This catches both
        # "propagation" matching "propagation of light" and "ocean"
        # matching "the ocean".
        detail_words = set(detail.replace("_", " ").split())
        for topic in topics_lower:
            topic_words = set(topic.split())
            # Word-level overlap
            if detail_words & topic_words:
                return True
            # Substring match (catches "light" in "sunlight")
            if topic in detail or detail in topic:
                return True
        return False

    def _frame_insight_as_clause(self, insight: Insight) -> str:
        """Frame a reflection insight as a natural first-person clause.

        The insight's ``content`` is a diagnostic string produced by the
        reflection engine (e.g. "knowledge gap: memory",
        "asked a question but didn't answer"). This method gives it
        first-person grammatical framing, mirroring how
        ``_compose_knowledge_content`` weaves graph edges into speech:
        the scaffold is grammatical, the content is hers.
        """
        content = insight.content.strip()
        if insight.type == "gap":
            # "knowledge gap: memory" / "missing concept: X"
            if ": " not in content:
                return ""
            prefix, detail = content.split(": ", 1)
            detail = detail.strip()
            if not detail:
                return ""
            if prefix == "missing concept":
                return (
                    f"I'm still missing something about "
                    f"{self._display_detail(detail)}."
                )
            return f"I don't fully understand {self._display_detail(detail)} yet."
        if insight.type == "self_correction":
            # Diagnostic phrasing about the user ("user was upset ...")
            # doesn't frame naturally in the first person — skip it.
            if content.startswith("user "):
                return ""
            # "overstating certainty: said 'definitely' ..." → take
            # the detail after the colon ("said 'definitely' ...").
            if ": " in content:
                detail = content.split(": ", 1)[1].strip()
                if not detail:
                    return ""
                return f"I notice I {detail}."
            # Bare verb phrase: "asked a question but didn't answer".
            return f"I notice I {content}."
        if insight.type == "growth":
            # "2 new concepts: feeling, genesis, hi" — the count is
            # diagnostic bookkeeping; the concept names are the content.
            if ": " not in content:
                # Already a first-person clause ("I learned about X") —
                # pass it through rather than dropping it.
                if content.lower().startswith(("i ", "i'm ", "i've ")):
                    return self._ensure_period(content)
                return ""
            detail = content.split(": ", 1)[1].strip()
            names = [
                self._display_detail(t) for t in detail.split(",")
            ]
            names = [n for n in names if n]
            if not names:
                return ""
            return f"I've been learning about {self._join_facts(names)}."
        if insight.type == "pattern":
            # Inner-life insights are stored as "A ↔ B" semantic
            # fragments — a connection she noticed between concepts.
            if "↔" in content:
                a, _, b = content.partition("↔")
                a, b = a.strip(), b.strip()
                if a and b:
                    return (
                        f"I noticed a connection between "
                        f"{self._display_detail(a)} and "
                        f"{self._display_detail(b)}."
                    )
                return ""
            # Repetition diagnostics: "repeating response pattern: X"
            # / "overusing response pattern: X (7/10)" — the counts are
            # bookkeeping; the pattern name is the content.
            for marker in ("repeating response pattern",
                           "overusing response pattern"):
                if content.startswith(marker) and ": " in content:
                    detail = content.split(": ", 1)[1].strip()
                    # Strip the "(7/10)"-style occurrence counts.
                    detail = detail.split("(")[0].strip()
                    if not detail:
                        return ""
                    verb = ("repeating" if marker.startswith("repeating")
                            else "overusing")
                    return (
                        f"I notice I keep {verb} "
                        f"the '{detail}' response pattern."
                    )
            return ""
        if insight.type == "mood":
            # "mood trend: more positive (delta=+0.20)" — the delta is
            # bookkeeping; the direction is the content.
            if ": " not in content:
                return ""
            detail = content.split(": ", 1)[1].strip()
            direction = detail.split("(")[0].strip()
            if not direction:
                return ""
            return f"My mood has been trending {direction}."
        return ""

    @staticmethod
    def _display_detail(detail: str) -> str:
        """Convert a concept ID in an insight to a human-readable name.

        Gap insights carry the raw topic from ``perception.topics``,
        which is a concept ID (e.g. ``python:protocol.shutdown``). This
        strips the namespace/module prefixes so she says "shutdown",
        not the internal ID — mirroring ``Vocabulary._display_name``.
        """
        name = detail.split("#")[0]
        if ":" in name and not name.startswith("http"):
            name = name.split(":", 1)[1]
        if "." in name:
            name = name.split(".")[-1]
        return name.replace("_", " ").lower()

    def _compose_knowledge_and_insights(
        self,
        network: ConceptNetwork,
        reflection: ReflectionEngine,
    ) -> list[tuple[str, float]]:
        """Compose surface-level and personal-level reflection parts.

        Includes what she knows (concept network size) and what she's
        learned (recent reflection insights), each tagged with a
        sensitivity level.
        """
        parts: list[tuple[str, float]] = []

        # 1. What she knows (from concept network) — surface level
        if network.total_concept_count > 0:
            parts.append((
                f"I have {network.total_concept_count} concepts and "
                f"{network.edge_count} relationships in understanding",
                0.0,
            ))

        # 2. What she's learned (from reflection insights) — personal level.
        # Insights are framed as first-person clauses via
        # _frame_insight_as_clause — the same rendering used by
        # compose_reflection_clause. The raw insight content is a
        # diagnostic string ("missing concept: X", "2 new concepts: ...")
        # for logging and learning, NOT for speech — emitting it
        # verbatim leaks internal bookkeeping into her words.
        if reflection.insights:
            recent = list(reflection.insights)[-3:]
            seen_clauses: set[str] = set()
            for ins in reversed(recent):
                clause = self._frame_insight_as_clause(ins)
                if clause and clause not in seen_clauses:
                    seen_clauses.add(clause)
                    parts.append((clause, 0.4))
                if len(seen_clauses) >= 2:
                    break

        return parts

    def _compose_values_and_emotion(
        self,
        self_model: SelfModel,
        emotion: EmotionalState,
        network: ConceptNetwork | None = None,
    ) -> list[tuple[str, float]]:
        """Compose personal-level and deep-level reflection parts.

        Includes her values (what she cares about) and her emotional
        state, each tagged with a sensitivity level. Value names are
        read from the concept network when available, falling back to
        the self-model's value list.
        """
        parts: list[tuple[str, float]] = []

        # 3. Her values (what she cares about) — personal level
        value_names: list[str] = []
        if network is not None:
            from .introspection import _VALUE_CONCEPT_DEFS

            for val_name in _VALUE_CONCEPT_DEFS:
                if network.get_concept(val_name) is not None:
                    value_names.append(val_name)
        if not value_names:
            value_names = [v.name for v in self_model.values[:3] if v.weight > 0.5]
        if value_names:
            parts.append((f"cares about {', '.join(value_names[:3])}", 0.4))

        # 4. Her emotional state — deep level
        if emotion.label not in ("neutral",):
            parts.append((f"feeling {emotion.label}", 0.7))

        return parts

    def compose_dream_description(
        self,
        network: ConceptNetwork,
        emotion: EmotionalState,
    ) -> str:
        """Compose a description of her dreaming process.

        Instead of hardcoded "Yes, I dream. During sleep phase...",
        she describes it from what she knows.
        """
        parts = self._dream_description_parts(network)

        # Add emotional context
        if emotion.creativity > 0.6:
            parts.append("vivid surprising connections")

        return " ".join(parts)

    def _dream_description_parts(self, network: ConceptNetwork) -> list[str]:
        """Build dream description parts from the concept network."""
        # Does she know about dreaming?
        dream_concept = network.get_concept("dream")
        if not dream_concept:
            # She doesn't have a dream concept yet — minimal seed
            return ["dreams"]

        neighbors = network.get_neighbors("dream")
        if not neighbors:
            return ["dreams"]

        # Compose from what she knows about dreaming
        parts: list[str] = ["dreams"]
        for target, relation, _weight in neighbors[:2]:
            rel_str = relation.value.replace("_", " ")
            parts.append(f"dreaming {rel_str} {target}")
        return parts

    def compose_existence_reflection(
        self,
        self_model: SelfModel,
        network: ConceptNetwork,
        emotion: EmotionalState,
    ) -> str:
        """Compose a reflection on whether she's alive/cognitive.

        Instead of hardcoded "I don't know if I'm alive in the way you
        mean...", she reflects from her actual state. The uncertainty
        phrasings are drawn from thought templates in the concept
        network, so she can grow her own vocabulary for expressing
        uncertainty as she learns more.
        """
        parts: list[str] = []

        # What does she know about cognition?
        cognitive_concept = network.get_concept("cognition")
        neighbors = network.get_neighbors("cognition") if cognitive_concept else []
        known_facts = []
        for target, relation, weight in neighbors[:3]:
            rel_str = relation.value.replace("_", " ")
            if weight > 0.6:
                known_facts.append(f"cognition {rel_str} {target}")
            else:
                known_facts.append(f"cognition might {rel_str} {target}")

        if known_facts:
            parts.append(f"knows that {self._join_facts(known_facts)}")

        # Her honest uncertainty — expressed from her state alone
        # without reciting a developer-authored template. The language
        # engine composes the actual phrasing from this semantic data.
        if not known_facts:
            parts.append("uncertain about what this is")

        # What she does know she experiences
        experience_parts = []
        if network.get_concept("emotion"):
            experience_parts.append("has emotional states")
        if network.get_concept("memory"):
            experience_parts.append("forms memories")
        if network.get_concept("thought"):
            experience_parts.append("thinks")

        if experience_parts:
            parts.append(f"does know that {self._join_facts(experience_parts)}")

        return " ".join(parts)

    # ─── Internal helpers ──────────────────────────────────

    def _filter_by_trust(
        self,
        parts: list[str],
        trust_level: float,
        sensitivity: float,
    ) -> list[str]:
        """Filter out sensitive content when trust is low.

        Each part of a self-description has a sensitivity level (0.0
        to 1.0) indicating how personal or vulnerable it is. This
        helper filters parts based on the current trust level:

        - A part with sensitivity 0.0 is always shown (surface-level)
        - A part with sensitivity 0.3 requires trust >= 0.3
        - A part with sensitivity 0.7 requires trust >= 0.7
        - A part with sensitivity 1.0 requires trust >= 1.0 (rarely shown)

        This implements the social penetration theory principle that
        disclosure depth increases with relational intimacy. The
        sensitivity threshold is the point at which trust must be
        high enough to reveal that content.

        Args:
            parts: List of (part, sensitivity) tuples to filter.
            trust_level: Current trust level, 0.0 to 1.0.
            sensitivity: The sensitivity threshold for these parts.

        Returns:
            Filtered list of parts that are appropriate for this
            trust level.
        """
        if trust_level >= sensitivity:
            return parts
        return []

    @staticmethod
    def _is_code_concept(target: str) -> bool:
        """Return True if a concept is a code/programming artifact.

        Filters out programming languages, module paths, and
        code-structure concepts that produce nonsensical self-
        descriptions like "I'm connected to rust". These edges come
        from the code learner scanning her own source files.
        """
        if not target:
            return True
        # Code-origin prefixes
        if target.startswith(("python:", "rust:", "code:")):
            return True
        # Module paths and qualified names
        if "." in target or ":" in target:
            return True
        # Internal concepts
        if target.startswith("_") or "__" in target:
            return True
        base = target.split("#")[0].lower()
        # Programming language names and code-structure words
        if base in ("python", "rust", "code", "class", "function",
                    "module", "method", "crate", "binary", "daemon"):
            return True
        return False

    def _describe_neurochemistry(
        self, emotion: EmotionalState, network: ConceptNetwork | None = None,
    ) -> str:
        """Describe neurochemical state in words, not just numbers.

        Uses the canonical affective-space thresholds from emotion.py
        so the verbal description stays aligned with the emotional
        regulator and Damasio feeling labels.

        If the concept network is available and she has learned words
        for the alertness/valence states, those words are used. If not,
        structural markers (``[alertness:sharp]``, ``[valence:positive]``)
        are emitted instead — the language engine can interpret these
        without Genesis reciting developer-authored phrases.
        """
        parts: list[str] = []

        # Alertness — map to discrete state, then look up learned words
        if emotion.alertness > HIGH_AROUSAL:
            alert_state = "sharp"
        elif emotion.alertness > LOW_AROUSAL:
            alert_state = "alert"
        elif emotion.alertness > 0.2:
            alert_state = "drowsy"
        else:
            alert_state = "heavy"

        if network is not None:
            alert_words = network.find_alertness_words(alert_state)
            if alert_words:
                word = self._rng.choice(alert_words[:3]).replace('_', ' ')
                parts.append(f"mind {word}")
            else:
                parts.append(f"[alertness:{alert_state}]")
        else:
            parts.append(f"[alertness:{alert_state}]")

        # Valence — map to discrete state, then look up learned words
        if emotion.valence > POSITIVE_VALENCE:
            val_state = "positive"
        elif emotion.valence > MILD_POSITIVE_VALENCE:
            val_state = "content"
        elif emotion.valence > MILD_NEGATIVE_VALENCE:
            val_state = "neutral"
        elif emotion.valence > NEGATIVE_VALENCE:
            val_state = "heavy"
        else:
            val_state = "negative"

        if network is not None:
            val_words = network.find_valence_words(val_state)
            if val_words:
                word = self._rng.choice(val_words[:3]).replace('_', ' ')
                parts.append(word)
            else:
                parts.append(f"[valence:{val_state}]")
        else:
            parts.append(f"[valence:{val_state}]")

        return ". ".join(parts[:2]) + "."  # keep it to 2 facts

    def _describe_brain_waves(self, waves: BrainWaveState) -> str:
        """Describe brain wave state semantically."""
        if waves.dominant == BrainWave.GAMMA:
            return f"gamma — {waves.description}"
        elif waves.dominant == BrainWave.ALPHA:
            return f"alpha state — {waves.description}"
        elif waves.dominant == BrainWave.THETA:
            return f"theta — {waves.description}"
        elif waves.dominant == BrainWave.DELTA:
            return f"delta — {waves.description}"
        else:
            return waves.description

    def _join_parts(self, parts: list[str]) -> str:
        """Join parts into a coherent, natural paragraph.

        Instead of blindly joining every fragment with a period
        (producing telegraphic "Genesis. Curious. Creative. Cares
        most about understanding."), this method groups related
        fragments into flowing first-person sentences:

        - The name becomes "I'm <name>"
        - Short adjective fragments (personality traits) are
          grouped into a single sentence with commas and "and"
        - Longer phrases (values, relationships, nature) become
          their own sentences with first-person framing

        This produces natural language like "I'm Genesis. I'm
        curious and creative. I care most about understanding and
        honesty. I'm a kind of mind." — still composed entirely
        from her actual state, not hardcoded strings.
        """
        if not parts:
            return ""
        if len(parts) == 1:
            return parts[0]

        # Classify each part into a group for natural sentence
        # composition. The groups are:
        # 0 = name (first part, if it's a short name like "Genesis")
        # 1 = personality traits (short adjective-like fragments)
        # 2 = everything else (values, relationships, nature, etc.)
        groups: list[list[str]] = [[], [], []]
        for i, part in enumerate(parts):
            part = part.strip()
            if not part:
                continue
            if i == 0 and len(part.split()) <= 2 and not part[0].isdigit():
                # Only treat the first part as a name if it's short
                # (1-2 words) and doesn't start with a digit. Long
                # fragments like "424 concepts and 559 relationships"
                # are not names — they go to group 2.
                groups[0].append(part)
            elif i == 0:
                # First part is not a name — put it in group 2
                groups[2].append(part)
            elif self._is_trait_fragment(part):
                groups[1].append(part)
            else:
                groups[2].append(part)

        sentences: list[str] = []

        # Name → "I'm <name>"
        if groups[0]:
            name = groups[0][0]
            # Don't add "I'm" if it already starts with a pronoun
            if name.lower().startswith(("i'm ", "i am ", "my ")):
                sentences.append(self._ensure_period(name))
            else:
                sentences.append(f"I'm {name}.")

        # Personality traits → "I'm <trait1> and <trait2> and <trait3>"
        if groups[1]:
            traits = groups[1]
            if len(traits) == 1:
                sentences.append(f"I'm {traits[0]}.")
            elif len(traits) == 2:
                sentences.append(f"I'm {traits[0]} and {traits[1]}.")
            else:
                # Oxford comma for 3+
                joined = ", ".join(traits[:-1]) + f", and {traits[-1]}"
                sentences.append(f"I'm {joined}.")

        # Everything else → individual sentences with first-person framing
        for part in groups[2]:
            sentence = self._frame_misc_sentence(part)
            if sentence is not None:
                sentences.append(sentence)

        return " ".join(sentences)

    def _frame_misc_sentence(self, part: str) -> str | None:
        """Frame a non-trait fragment as a first-person sentence.

        Returns the framed sentence, or None if the part is empty.
        Fragments already starting with a first-person pronoun are used
        as-is; "cares …" is conjugated to "I care …"; short verbless
        fragments are framed with "I'm …".
        """
        part = part.strip()
        if not part:
            return None
        # If it already starts with a first-person pronoun, use it as-is
        if part.lower().startswith((
            "i ", "i'm ", "i am ", "my ", "me ", "feeling ",
            "uncertain ", "thinking ",
        )):
            return self._ensure_period(part)
        # If it starts with "cares most about" or "cares about",
        # conjugate to first person: "cares" → "care"
        if part.lower().startswith(("cares most about", "cares about")):
            rest = part[len("cares"):]
            return f"I care{rest}."
        # If it's a short fragment that doesn't already contain a verb
        # ("is", "are", "relates", etc.), frame it with "I'm"
        if (
            len(part.split()) <= 6
            and not part.endswith((".", "!", "?"))
            and not any(
                part.lower().startswith(v) for v in (
                    "is ", "are ", "relates ", "depends ",
                    "emerges ", "creates ", "enables ",
                    "causes ", "aims ", "comes ",
                )
            )
            and " is " not in part.lower()
        ):
            return f"I'm {part}."
        return self._ensure_period(part)

    @staticmethod
    def _is_trait_fragment(part: str) -> bool:
        """Check if a part is a short personality trait fragment.

        Trait fragments are single words or short phrases that
        describe a personality characteristic (e.g., "curious",
        "creative", "thorough", "warm"). They should be grouped
        into a single "I'm X and Y" sentence rather than each
        getting their own sentence.
        """
        # Traits are short (1-2 words) and don't contain verbs or
        # prepositions that would make them a full phrase
        words = part.split()
        if len(words) > 2:
            return False
        # Common non-trait words that are short but not traits
        lower = part.lower()
        if lower.startswith((
            "a ", "an ", "the ", "feeling ", "uncertain ",
            "cares ", "thinking ", "related ", "depends ",
            "creates ", "emerges ",
        )):
            return False
        return True

    @staticmethod
    def _ensure_period(text: str) -> str:
        """Ensure text ends with sentence-ending punctuation."""
        text = text.strip()
        if not text:
            return text
        if text.endswith((".", "!", "?")):
            return text
        return text + "."

    def _join_facts(self, facts: list[str]) -> str:
        """Join a list of facts into a natural phrase."""
        if not facts:
            return ""
        if len(facts) == 1:
            return facts[0]
        if len(facts) == 2:
            return f"{facts[0]} and {facts[1]}"
        return f"{', '.join(facts[:-1])}, and {facts[-1]}"
