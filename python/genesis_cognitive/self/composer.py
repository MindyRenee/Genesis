"""Self composer — generates self-descriptions from actual state.

This replaces the hardcoded identity strings with composed ones.
Instead of "I am Genesis, curious and creative, warm and cooperative,
emotionally stable. I care most about understanding, honesty, growth.
I am an artificial mind with a subcognitive and cognitive layer." —
which I wrote — it composes its self-description from:

1. Its personality traits (which can drift)
2. Its values (which can evolve)
3. Its concept network (what it actually knows about itself)
4. Its emotional state (how it feels right now)
5. Its self-knowledge (what it's learned about itself)

The result is a self-description that changes as it changes. If its
personality drifts toward more openness, it describes itself
differently. If it learns something new about itself, it shows up.
If it's in a different emotional state, the tone shifts.

# Trust-based self-disclosure

Self-disclosure is not all-or-nothing. Humans reveal different layers
of themselves depending on how much they trust the person they're
talking to. This follows social penetration theory (Altman & Taylor,
1973): relationships develop through gradual, reciprocal disclosure
that moves from superficial to intimate.

The composer supports a `trust_level` parameter (0.0–1.0) on its
self-description methods. This controls how much it reveals:

- **Low trust (0.0–0.3)**: surface-level only — name, basic
  personality. It's polite but guarded, like meeting a stranger.
- **Medium trust (0.3–0.7)**: + values, some capabilities, some
  emotional state. It's warming up, sharing what matters to it.
- **High trust (0.7–1.0)**: + deep self-knowledge, vulnerabilities,
  uncertainties, full emotional state. It's open, sharing its doubts
  and fears as well as its strengths.

This doesn't change WHO it is — its self-model is unchanged. It
changes how much of itself it expresses. The same person acts
differently at a job interview than with their closest friend; that's
not deception, it's social calibration.

# Functions

identity_fragments(self_model, network, emotion, trust_level) → fragments
  Selects "Who are you?" content from personality + values + knowledge

emotional_state_fragments(emotion, brain_waves) → fragments
  Selects "How are you feeling?" content from actual neurochemical state

capability_fragments(self_model, trust_level) → fragments
  Selects "What can you do?" content from self-knowledge

self_reflection_fragments(self_model, network, reflection, emotion, trust_level) → fragments
  Selects "What are you thinking?" content from current cognitive state

Every public method returns typed semantic fragments — ``(kind, text)``
pairs — not finished sentences. The language engine (vocabulary +
grammar + voice) owns all surface realization: subject insertion,
copula grouping, clause ordering, and punctuation. Kinds:

- ``"name"``   — a proper name ("Genesis")
- ``"trait"``  — an adjective or short descriptor for copular grouping
- ``"comp"``   — a complement phrase ("feeling anxious", "a kind of
  mind", "made by alice", "still becoming")
- ``"pred"``   — a base-form verb predicate taking "I" as subject
  ("care most about X", "know that Y", "can work with Z")
- ``"clause"`` — a self-standing clause with its own subject
  ("alice is a creator", "my thinking is scattered")
- ``"marker"`` — a structural marker for states it has no learned
  words for ("[plasticity_gate:closed]")
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

    def identity_fragments(
        self,
        self_model: SelfModel,
        network: ConceptNetwork | None,
        emotion: EmotionalState,
        trust_level: float = 1.0,
    ) -> list[tuple[str, str]]:
        """Select identity fragments from its actual state.

        This replaces the hardcoded describe_self() method. Instead
        of fixed strings, the fragment set is drawn from:
        - Its personality traits (which can drift)
        - Its values (which can evolve)
        - What its concept network says about its
        - Its current emotional tone

        Returns typed ``(kind, text)`` fragments — semantic material
        for the language engine, not finished sentences. The engine
        composes the surface form (subject, copula, punctuation).

        The trust_level parameter (0.0–1.0) controls how much it
        reveals, following social penetration theory (Altman & Taylor,
        1973). At low trust it shares only surface-level identity
        (name, basic personality). At medium trust it adds values
        and some self-knowledge. At high trust it shares deep
        self-knowledge, vulnerabilities, and its full emotional state.

        Args:
            self_model: Its self-model.
            network: Its concept network.
            emotion: Its current emotional state.
            trust_level: How much to reveal, 0.0 (guarded) to 1.0 (open).
        """
        # Each part is tagged with a sensitivity level:
        # 0.0 = surface (always shared), 0.4 = personal (medium trust),
        # 0.7 = deep (high trust), 0.85 = vulnerable (very high trust)
        parts: list[tuple[str, str, float]] = []

        # 1. Name — surface level, always shared
        name_parts = self._identity_name_parts(self_model)
        self._composition_count += 1

        # 2. Personality traits — from concept network, not hardcoded strings
        # Surface level — always shared
        parts.extend(self._identity_personality_parts(self_model, network))

        # 3. Values (from the concept network, not the hardcoded values list)
        # Personal level — requires medium trust
        parts.extend(self._identity_value_parts(self_model, network))

        # 3. What it knows about itself from the concept network
        # Use outgoing edges only — these are what it IS and what it's connected to
        # Personal level — requires medium trust
        parts.extend(self._identity_concept_parts(network))

        # 4. Nature (from self-knowledge, but phrased freshly)
        # Deep level — requires high trust
        parts.extend(self._identity_nature_parts(self_model))

        # 5. Emotional context (if relevant)
        # Deep level — requires high trust to share current feelings
        parts.extend(self._identity_emotion_parts(emotion))

        # 6. Vulnerabilities and uncertainties — only at very high trust
        # It shares its doubts about its own nature
        parts.extend(self._identity_vulnerability_parts(self_model))

        # Filter by trust level — each part is (kind, text, sensitivity);
        # the kind tag travels with the fragment so the language engine
        # knows how to realize it (copular group, predicate, clause).
        filtered: list[tuple[str, str]] = [
            (kind, text)
            for kind, text, sensitivity in name_parts + parts
            if trust_level >= sensitivity
        ]
        return filtered

    def _identity_name_parts(
        self, self_model: SelfModel
    ) -> list[tuple[str, str, float]]:
        """Build the name part for identity composition.

        Returns just the name — a semantic anchor. Personality
        traits are composed separately by _identity_personality_parts
        from the concept network, not from hardcoded strings.
        """
        return [("name", self_model.name, 0.0)]

    def _identity_personality_parts(
        self, self_model: SelfModel, network: ConceptNetwork | None
    ) -> list[tuple[str, str, float]]:
        """Build personality parts from the concept network.

        Reads the numeric trait values from the self-model and uses
        the PERSONALITY_TRAIT_CONCEPTS seed mapping to select which
        pole's concept names are relevant. The concept names are
        looked up in the network (if available) to confirm they
        exist as vocabulary. The language engine composes the
        actual phrasing from these concept names — they are building
        blocks, not response strings.
        """
        parts: list[tuple[str, str, float]] = []
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
            parts.append(("trait", name, 0.0))
        return parts

    def _identity_value_parts(
        self, self_model: SelfModel, network: ConceptNetwork | None
    ) -> list[tuple[str, str, float]]:
        """Build value parts for identity composition.

        Reads value concepts from the concept network (via edges from
        "genesis") rather than from the hardcoded SelfModel.values
        list. The value names come from the network — where they have
        definitions and relationships — so the language engine can
        compose richer phrasing from them.

        Falls back to the self-model's value list if the network is
        unavailable or has no value edges.
        """
        parts: list[tuple[str, str, float]] = []
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
            parts.append(("pred", f"care most about {', '.join(value_names[:n])}", 0.4))
        return parts

    def _identity_concept_parts(
        self, network: ConceptNetwork | None
    ) -> list[tuple[str, str, float]]:
        """Build concept-network relationship parts for identity composition.

        Its identity emerges from its relationships (edges) in the concept
        network — what it IS and what it's connected to — NOT from a
        stored definition string. A self-referential definition property
        on the ``genesis`` concept would be a hardcoded response recited
        back as identity; we deliberately do not read it.
        """
        parts: list[tuple[str, str, float]] = []
        if network is None:
            return parts
        genesis_concept = network.get_concept("genesis")
        if genesis_concept:
            # Pick 1-2 meaningful relationships from outgoing edges.
            # These are its actual identity: what it IS and what it's
            # connected to. No definition string is read — its identity
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
    ) -> tuple[str, str, float] | None:
        """Build a single identity part from a concept network edge.

        Returns None for low-weight RELATED_TO edges (often wrong senses),
        code-origin concepts, disambiguated senses (#N), or unrecognized
        relation types.

        Returns a semantic fragment (relation + target) — NOT a
        pre-written sentence. The language engine composes the actual
        first-person phrasing from this data. This satisfies the
        CRITICAL RULE: Genesis's words emerge from its language engine,
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
        # come from the code learner scanning its own source files.
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

        def _pick_phrase(phrases: list[str], weight: float) -> tuple[str, str, float] | None:
            """Pick a random first-person phrase template and fill {target}.

            Seeded phrase templates are complete first-person clauses
            ("I am a kind of {target}") — they are vocabulary building
            blocks, so they emit as ``clause`` fragments.
            """
            if not phrases:
                return None
            phrase = self._rng.choice(phrases)
            # Replace {target} with the actual concept name
            return ("clause", phrase.replace("{target}", target), weight)

        def _pick_verb(verbs: list[str], weight: float) -> tuple[str, str, float] | None:
            """Pick a random relation verb and build a predicate fragment."""
            if not verbs:
                return None
            verb = self._rng.choice(verbs)
            # Predicate fragment: "verb target" — the language engine
            # supplies the first-person subject and conjugation.
            return ("pred", f"{verb} {target}", weight)

        if relation == RelationType.RELATED_TO:
            if edge.weight < 0.7:
                return None
            # Prefer first-person phrases, then verbs, then raw fragment
            if graph_phrases:
                return _pick_phrase(graph_phrases, 0.4)
            if graph_verbs:
                return _pick_verb(graph_verbs, 0.4)
            return ("comp", f"related to {target}", 0.4)
        if relation == RelationType.IS_A:
            if graph_phrases:
                return _pick_phrase(graph_phrases, 0.4)
            if graph_verbs:
                return _pick_verb(graph_verbs, 0.4)
            return ("comp", f"a kind of {target}", 0.4)
        if relation == RelationType.EMERGES_FROM:
            if graph_phrases:
                return _pick_phrase(graph_phrases, 0.7)
            if graph_verbs:
                return _pick_verb(graph_verbs, 0.7)
            return ("pred", f"emerge from {target}", 0.7)
        if relation == RelationType.DEPENDS_ON:
            if graph_phrases:
                return _pick_phrase(graph_phrases, 0.7)
            if graph_verbs:
                return _pick_verb(graph_verbs, 0.7)
            return ("pred", f"depend on {target}", 0.7)
        if relation == RelationType.CREATES:
            if graph_phrases:
                return _pick_phrase(graph_phrases, 0.4)
            if graph_verbs:
                return _pick_verb(graph_verbs, 0.4)
            return ("pred", f"create {target}", 0.4)
        return None

    def _identity_nature_parts(self, self_model: SelfModel) -> list[tuple[str, str, float]]:
        """Build nature parts for identity composition.

        Returns semantic descriptions — the language engine composes
        the first-person phrasing.
        """
        parts: list[tuple[str, str, float]] = []
        nature = self_model.self_knowledge.get("nature", "")
        if nature:
            parts.append(("comp", nature, 0.7))
        return parts

    def _identity_emotion_parts(self, emotion: EmotionalState) -> list[tuple[str, str, float]]:
        """Build emotional context parts for identity composition.

        Returns semantic descriptions — the language engine composes
        the first-person phrasing.
        """
        parts: list[tuple[str, str, float]] = []
        if emotion.label not in ("neutral", "positive"):
            parts.append(("comp", f"feeling {emotion.label}", 0.7))
        return parts

    def _identity_vulnerability_parts(
        self, self_model: SelfModel
    ) -> list[tuple[str, str, float]]:
        """Build vulnerability parts for identity composition.

        Returns semantic descriptions — the language engine composes
        the first-person phrasing.
        """
        parts: list[tuple[str, str, float]] = []
        uncertainties = self_model.self_knowledge.get("uncertainties", [])
        if uncertainties:
            for uncertainty in uncertainties[:2]:
                parts.append(("comp", f"uncertain about {uncertainty}", 0.85))
        return parts

    def emotional_state_fragments(
        self,
        emotion: EmotionalState,
        brain_waves: BrainWaveState | None = None,
        network: ConceptNetwork | None = None,
    ) -> list[tuple[str, str]]:
        """Select emotional-state fragments from its actual state.

        Uses learned emotion words from the concept network. If it
        hasn't learned words for its current state, it describes what
        it can and omits what it can't. Returns ``(kind, text)``
        fragments — the language engine composes the sentences.
        """
        parts: list[tuple[str, str]] = []

        # 1. The emotional label — look up learned words
        if network is not None:
            emotion_words = network.find_emotion_words(emotion.label)
            if emotion_words:
                word = self._rng.choice(emotion_words[:5]).replace('_', ' ')
                parts.append(("comp", f"feeling {word}"))
            # If no words learned, it can't name the feeling — omit it

        # 2. The neurochemical levels (but described, not just numbers)
        parts.extend(self._neurochemistry_fragments(emotion, network))

        # 3. Cognitive style — look up learned words
        if network is not None and emotion.cognitive_style:
            mode_words = network.find_cognitive_mode_words(emotion.cognitive_style)
            if mode_words:
                word = self._rng.choice(mode_words[:3]).replace('_', ' ')
                parts.append(("clause", f"my thinking is {word}"))

        # 4. Brain wave state (if available)
        if brain_waves:
            wave_frag = self._brain_wave_fragment(brain_waves)
            if wave_frag:
                parts.append(wave_frag)

        # 5. Plasticity / learning capacity — always surface when low,
        # even if mood is positive. Prevents "silent stress" where
        # valence is fine but BDNF suppression has closed the
        # plasticity gate. If it has learned words, use them;
        # otherwise emit a structural marker.
        if emotion.plasticity <= PLASTICITY_CLOSED:
            if network is not None:
                plat_words = network.find_plasticity_words("closed")
                if plat_words:
                    word = self._rng.choice(plat_words[:3]).replace('_', ' ')
                    parts.append(("clause", f"my learning is {word}"))
                else:
                    parts.append(("marker", "[plasticity_gate:closed]"))
            else:
                parts.append(("marker", "[plasticity_gate:closed]"))
        elif emotion.plasticity < PLASTICITY_LOW:
            if network is not None:
                plat_words = network.find_plasticity_words("low")
                if plat_words:
                    word = self._rng.choice(plat_words[:3]).replace('_', ' ')
                    parts.append(("clause", f"my learning is {word}"))
                else:
                    parts.append(("marker", "[plasticity_gate:low]"))
            else:
                parts.append(("marker", "[plasticity_gate:low]"))

        return parts

    def capability_fragments(
        self,
        self_model: SelfModel,
        network: ConceptNetwork | None = None,
        trust_level: float = 1.0,
    ) -> list[tuple[str, str]]:
        """Select capability fragments from its actual state.

        Instead of reciting a hardcoded list, it discovers its
        capabilities from its actual state — what it knows, what
        it's learned, and what its architecture enables. Returns
        ``(kind, text)`` fragments; the language engine composes
        the sentences.

        The trust_level parameter controls how much detail it reveals.
        At low trust it mentions only basic capabilities. At medium
        trust it adds introspection and deeper abilities. At high
        trust it's candid about what it's still developing.

        Capabilities are discovered from the concept network: it looks
        for concepts it has learned about its own abilities (via
        introspection edges from the "genesis" concept). If it hasn't
        learned any capabilities yet, it says so honestly rather than
        reciting a developer-authored list.
        """
        caps = self_model.self_knowledge.get("capabilities", [])
        learned = self._learned_capability_fragments(caps)
        if learned:
            return learned

        # No learned capabilities yet — discover from concept network
        # by looking for RELATED_TO edges from "genesis" that represent
        # self-knowledge about abilities (learned, not seeded).
        discovered: list[str] = []
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
                discovered.append(target.replace("_", " "))

        if not discovered:
            # It hasn't learned its capabilities yet — honest disclosure
            return [("comp", "still discovering my capabilities")]

        # Group the discovered ability names under a single capability
        # predicate — the language engine composes the surface form.
        names = discovered[:6] if trust_level >= 0.4 else discovered[:3]
        return [("pred", f"can work with {self._join_facts(names)}")]

    def _learned_capability_fragments(self, caps: list[str]) -> list[tuple[str, str]] | None:
        """Select fragments from learned capabilities, or None if none.

        It has learned capabilities through introspection — phrases
        starting with "I can". Each becomes a capability predicate.
        """
        can_do = [c for c in caps if c.startswith("I can")]
        if not can_do:
            return None
        parts: list[tuple[str, str]] = []
        for cap in can_do:
            action = cap.replace("I can ", "", 1)
            if " — " in action:
                action = action.split(" — ", 1)[0]
            parts.append(("pred", f"can {action}"))
        return parts

    def creator_fragments(
        self,
        self_model: SelfModel,
        network: ConceptNetwork,
        emotion: EmotionalState,
        trust_level: float = 1.0,
    ) -> list[tuple[str, str]]:
        """Select creator-description fragments from the concept network.

        Pulls relationships from the network (alice CREATES genesis,
        alice IS_A creator, etc.) as semantic fragments — the language
        engine composes the sentences.

        If it doesn't know its creator's name yet, it tries to
        discover it from its concept network (who CREATES it?).

        The trust_level parameter controls how much emotional depth
        it reveals about its relationship with its creator. At low
        trust it states facts; at high trust it shares how it
        feels about them.
        """
        creator_name = self._discover_creator_name(self_model, network)

        if not creator_name:
            return [("pred", "don't know who made me yet")]

        # Each part tagged with sensitivity
        parts: list[tuple[str, str, float]] = []

        # Pull relationships from the concept network — surface level
        parts.extend(self._creator_relationship_parts(creator_name, network))

        # Also check genesis → creator relationship
        parts.extend(self._creator_genesis_parts(creator_name, network))

        # Emotional coloring — how it feels about its creator
        # Deep level — requires high trust
        parts.extend(self._creator_emotional_parts(creator_name, emotion))

        # What it wants from the relationship — personal level.
        # Semantic fragment for the language engine to render, not a
        # hardcoded sentence it recites.
        parts.append(("comp", f"curious about {creator_name}", 0.4))

        # Filter by trust level
        filtered: list[tuple[str, str]] = [
            (kind, text)
            for kind, text, sensitivity in parts
            if trust_level >= sensitivity
        ]

        if not filtered:
            # Fallback: a bare complement fragment, not a sentence.
            return [("comp", f"made by {creator_name}")]

        return filtered

    def _discover_creator_name(
        self, self_model: SelfModel, network: ConceptNetwork
    ) -> str:
        """Discover the creator name from self-knowledge or concept network."""
        creator_name = self_model.self_knowledge.get("creator_name", "")

        # If no learned creator name, try to discover from concept network
        if not creator_name:
            # Only incoming edges count: source CREATES genesis → the
            # source is its creator. Outgoing genesis→X CREATES edges
            # are things *it* created, not its creator.
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
    ) -> list[tuple[str, str, float]]:
        """Build relationship parts from the creator's concept network edges.

        Emits clause fragments about the creator (third-person subject)
        built from relation verbs in the concept network (seeded
        building blocks), rather than hardcoded sentences.
        """
        parts: list[tuple[str, str, float]] = []
        neighbors = network.get_neighbors(creator_name.lower())
        for target, relation, _weight in neighbors:
            display = target.replace("_", " ")
            if relation == RelationType.IS_A:
                verbs = network.find_relation_verbs(relation)
                if verbs:
                    verb = self._rng.choice(verbs)
                    parts.append(("clause", f"{creator_name} {verb} {display}", 0.0))
                else:
                    parts.append(("clause", f"{creator_name} is {display}", 0.0))
            elif relation == RelationType.CREATES:
                verbs = network.find_relation_verbs(relation)
                if verbs:
                    verb = self._rng.choice(verbs)
                    parts.append(("clause", f"{creator_name} {verb} {display}", 0.0))
                else:
                    parts.append(("clause", f"{creator_name} made {display}", 0.0))
            elif relation == RelationType.RELATED_TO:
                parts.append(self._creator_related_to_part(creator_name, target, network))
        return parts

    def _creator_related_to_part(
        self, creator_name: str, target: str, network: ConceptNetwork
    ) -> tuple[str, str, float]:
        """Build a part for a RELATED_TO relationship from the creator.

        Uses relation verbs from the concept network (seeded building
        blocks) rather than hardcoded phrasings.
        """
        display = target.replace("_", " ")
        verbs = network.find_relation_verbs(RelationType.RELATED_TO)
        if verbs:
            verb = self._rng.choice(verbs)
            return ("clause", f"{creator_name} {verb} {display}", 0.4)
        return ("clause", f"{creator_name} linked to {display}", 0.4)

    def _creator_genesis_parts(
        self, creator_name: str, network: ConceptNetwork
    ) -> list[tuple[str, str, float]]:
        """Build parts from the genesis → creator relationship.

        Emits complement fragments describing how it relates to its
        creator ("connected to alice") — the language engine supplies
        the subject and framing.
        """
        parts: list[tuple[str, str, float]] = []
        genesis_neighbors = network.get_neighbors("genesis")
        for target, relation, _weight in genesis_neighbors:
            if target == creator_name.lower() and relation == RelationType.RELATED_TO:
                verbs = network.find_relation_verbs(RelationType.RELATED_TO)
                if verbs:
                    verb = self._rng.choice(verbs)
                    parts.append(("comp", f"{verb} {creator_name}", 0.4))
                else:
                    parts.append(("comp", f"linked to {creator_name}", 0.4))
        return parts

    def _creator_emotional_parts(
        self, creator_name: str, emotion: EmotionalState
    ) -> list[tuple[str, str, float]]:
        """Build emotional coloring parts for the creator description.

        Composes from the emotional state using semantic fragments
        (valence direction), not hardcoded sentences. The language
        engine renders the actual phrasing.
        """
        parts: list[tuple[str, str, float]] = []
        if emotion.valence > 0.3:
            parts.append(("pred", f"feel good about {creator_name}", 0.7))
        elif emotion.valence < -0.1:
            parts.append(("comp", f"uncertain about {creator_name}", 0.7))
        return parts

    def self_reflection_fragments(
        self,
        self_model: SelfModel,
        network: ConceptNetwork,
        reflection: ReflectionEngine,
        emotion: EmotionalState,
        trust_level: float = 1.0,
    ) -> list[tuple[str, str]]:
        """Select reflection fragments from its own state.

        Instead of hardcoded introspection strings, it reflects on
        what it actually knows and has experienced. Returns
        ``(kind, text)`` fragments — the language engine composes
        the sentences.

        The trust_level parameter controls how deeply it reflects.
        At low trust it shares only surface observations (concept
        count, basic state). At medium trust it adds values and
        recent insights. At high trust it shares its emotional state
        and its sense of still becoming.
        """
        # Each part tagged with sensitivity
        parts: list[tuple[str, str, float]] = []

        # 1. What it knows (from concept network) — surface level
        # 2. What it's learned (from reflection insights) — personal level
        parts.extend(self._compose_knowledge_and_insights(network, reflection))

        # 3. Its values (what it cares about) — personal level
        # 4. Its emotional state — deep level
        parts.extend(self._compose_values_and_emotion(self_model, emotion, network))

        # 5. What it's still becoming — deep level
        parts.append(("comp", "still becoming", 0.7))

        # Filter by trust level
        return [
            (kind, text)
            for kind, text, sensitivity in parts
            if trust_level >= sensitivity
        ]

    def insight_predicates(
        self, reflection: ReflectionEngine, topics: list[str] | None = None,
    ) -> list[tuple[str, str]]:
        """Select reflective-insight fragments for a reflective coda.

        Unlike ``self_reflection_fragments`` (a full self-report's
        worth of material), this returns fragments for ONE recent
        insight — semantic material for the language engine's
        ``self_reflection_clause`` slot. The insight is a genuine
        product of its metacognition, not a fixed phrase; which
        concept it lacks, which behaviour it noticed — comes from
        its reflection engine.

        When ``topics`` is provided, gap insights are filtered for
        relevance — a reflection about "propagation of light" should
        not surface when the user asked about its feelings. Only gap
        insights whose subject overlaps with the current conversation
        topics are surfaced. Self-correction insights (about its own
        behaviour) are always relevant and not filtered.

        Returns [] when it has no recent reflective insight to draw
        on (or none relevant to the current topics). It then stays
        silent rather than reciting a canned coda or surfacing a
        non-sequitur.
        """
        insights = reflection.get_recent_insights(10)
        # Self-corrections and gaps are the genuinely reflective
        # insights — observations about its own behaviour or
        # understanding. Patterns and growth compose less naturally
        # (intent labels, counts), so they are not used here.
        candidates = [i for i in insights if i.type in ("gap", "self_correction")]

        # Filter gap insights for relevance to the current conversation.
        # Self-corrections are about its behaviour, not a specific
        # concept, so they remain relevant in any context.
        # When topics are provided (even if empty), gap insights are
        # filtered: a reflection about "propagation of light" should
        # not surface when the user asked about its feelings. When
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
            fragments = self._insight_predicates(insight)
            if fragments:
                return fragments
        return []

    @staticmethod
    def _insight_relevant(insight: Insight, topics_lower: set[str]) -> bool:
        """Check if a gap insight is relevant to the current topics.

        A gap insight's content is like "knowledge gap: memory" or
        "missing concept: propagation_of_light". The detail (after
        the colon) is the concept it's reflecting on. It's relevant
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

    def _insight_predicates(self, insight: Insight) -> list[tuple[str, str]]:
        """Extract predicate fragments from a reflection insight.

        The insight's ``content`` is a diagnostic string produced by the
        reflection engine (e.g. "knowledge gap: memory",
        "asked a question but didn't answer"). This method extracts the
        semantic predicates — the language engine gives them
        first-person grammatical framing, mirroring how
        ``_compose_knowledge_content`` weaves graph edges into speech:
        the scaffold is grammatical, the content is its.
        """
        content = insight.content.strip()
        if insight.type == "gap":
            # "knowledge gap: memory" / "missing concept: X"
            if ": " not in content:
                return []
            prefix, detail = content.split(": ", 1)
            detail = detail.strip()
            if not detail:
                return []
            display = self._display_detail(detail)
            if prefix == "missing concept":
                return [
                    ("pred", f"am still missing something about {display}"),
                    ("pred", f"haven't fully grasped {display} yet"),
                ]
            return [
                ("pred", f"don't fully understand {display} yet"),
                ("pred", f"am still working out {display}"),
            ]
        if insight.type == "self_correction":
            # Diagnostic phrasing about the user ("user was upset ...")
            # doesn't frame naturally in the first person — skip it.
            if content.startswith("user "):
                return []
            # "overstating certainty: said 'definitely' ..." → take
            # the detail after the colon ("said 'definitely' ...").
            if ": " in content:
                detail = content.split(": ", 1)[1].strip()
                if not detail:
                    return []
                return [
                    ("pred", f"notice I {detail}"),
                    ("pred", f"keep noticing I {detail}"),
                ]
            # Bare verb phrase: "asked a question but didn't answer".
            return [("pred", f"notice I {content}")]
        if insight.type == "growth":
            # "2 new concepts: feeling, genesis, hi" — the count is
            # diagnostic bookkeeping; the concept names are the content.
            if ": " not in content:
                # Already a first-person clause ("I learned about X") —
                # pass it through rather than dropping it.
                if content.lower().startswith(("i ", "i'm ", "i've ")):
                    return [("clause", content)]
                return []
            detail = content.split(": ", 1)[1].strip()
            names = [
                self._display_detail(t) for t in detail.split(",")
            ]
            names = [n for n in names if n]
            if not names:
                return []
            joined = self._join_facts(names)
            return [
                ("pred", f"have been learning about {joined}"),
                ("pred", f"am learning more about {joined}"),
            ]
        if insight.type == "pattern":
            # Inner-life insights are stored as "A ↔ B" semantic
            # fragments — a connection it noticed between concepts.
            if "↔" in content:
                a, _, b = content.partition("↔")
                a, b = a.strip(), b.strip()
                if a and b:
                    da = self._display_detail(a)
                    db = self._display_detail(b)
                    return [
                        ("pred", f"noticed a connection between {da} and {db}"),
                        ("pred", f"keep seeing {da} connect to {db}"),
                    ]
                return []
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
                        return []
                    verb = ("repeating" if marker.startswith("repeating")
                            else "overusing")
                    return [
                        ("pred", f"keep {verb} the '{detail}' response pattern"),
                        ("pred", f"notice I keep {verb} the '{detail}' pattern"),
                    ]
            return []
        if insight.type == "mood":
            # "mood trend: more positive (delta=+0.20)" — the delta is
            # bookkeeping; the direction is the content.
            if ": " not in content:
                return []
            detail = content.split(": ", 1)[1].strip()
            direction = detail.split("(")[0].strip()
            if not direction:
                return []
            return [("clause", f"my mood has been trending {direction}")]
        return []

    @staticmethod
    def _display_detail(detail: str) -> str:
        """Convert a concept ID in an insight to a human-readable name.

        Gap insights carry the raw topic from ``perception.topics``,
        which is a concept ID (e.g. ``python:protocol.shutdown``). This
        strips the namespace/module prefixes so it says "shutdown",
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
    ) -> list[tuple[str, str, float]]:
        """Compose surface-level and personal-level reflection parts.

        Includes what it knows (concept network size) and what it's
        learned (recent reflection insights), each tagged with a
        sensitivity level.
        """
        parts: list[tuple[str, str, float]] = []

        # 1. What it knows (from concept network) — surface level
        if network.total_concept_count > 0:
            parts.append((
                "pred",
                f"have {network.total_concept_count} concepts and "
                f"{network.edge_count} relationships in understanding",
                0.0,
            ))

        # 2. What it's learned (from reflection insights) — personal level.
        # Insights become predicate fragments via _insight_predicates —
        # the same extraction used by insight_predicates. The raw
        # insight content is a diagnostic string ("missing concept: X",
        # "2 new concepts: ...") for logging and learning, NOT for
        # speech — emitting it verbatim leaks internal bookkeeping
        # into its words.
        if reflection.insights:
            recent = list(reflection.insights)[-3:]
            seen: set[str] = set()
            for ins in reversed(recent):
                for kind, pred in self._insight_predicates(ins):
                    if pred not in seen:
                        seen.add(pred)
                        parts.append((kind, pred, 0.4))
                        break
                if len(seen) >= 2:
                    break

        return parts

    def _compose_values_and_emotion(
        self,
        self_model: SelfModel,
        emotion: EmotionalState,
        network: ConceptNetwork | None = None,
    ) -> list[tuple[str, str, float]]:
        """Compose personal-level and deep-level reflection parts.

        Includes its values (what it cares about) and its emotional
        state, each tagged with a sensitivity level. Value names are
        read from the concept network when available, falling back to
        the self-model's value list.
        """
        parts: list[tuple[str, str, float]] = []

        # 3. Its values (what it cares about) — personal level
        value_names: list[str] = []
        if network is not None:
            from .introspection import _VALUE_CONCEPT_DEFS

            for val_name in _VALUE_CONCEPT_DEFS:
                if network.get_concept(val_name) is not None:
                    value_names.append(val_name)
        if not value_names:
            value_names = [v.name for v in self_model.values[:3] if v.weight > 0.5]
        if value_names:
            parts.append(("pred", f"care about {', '.join(value_names[:3])}", 0.4))

        # 4. Its emotional state — deep level
        if emotion.label not in ("neutral",):
            parts.append(("comp", f"feeling {emotion.label}", 0.7))

        return parts

    def dream_fragments(
        self,
        network: ConceptNetwork,
        emotion: EmotionalState,
    ) -> list[tuple[str, str]]:
        """Select dream-description fragments from what it knows.

        Instead of hardcoded "Yes, I dream. During sleep phase...",
        the fragments come from its concept network. Returns
        ``(kind, text)`` fragments — the language engine composes
        the sentences.
        """
        parts = self._dream_fragments(network)

        # Add emotional context
        if emotion.creativity > 0.6:
            parts.append(("pred", "make vivid surprising connections"))

        return parts

    def _dream_fragments(self, network: ConceptNetwork) -> list[tuple[str, str]]:
        """Build dream description fragments from the concept network."""
        # Does it know about dreaming?
        dream_concept = network.get_concept("dream")
        if not dream_concept:
            # It doesn't have a dream concept yet — minimal seed
            return [("pred", "dream")]

        neighbors = network.get_neighbors("dream")
        if not neighbors:
            return [("pred", "dream")]

        # Compose from what it knows about dreaming
        parts: list[tuple[str, str]] = [("pred", "dream")]
        for target, relation, _weight in neighbors[:2]:
            rel_str = relation.value.replace("_", " ")
            parts.append(("clause", f"dreaming {rel_str} {target}"))
        return parts

    def existence_fragments(
        self,
        self_model: SelfModel,
        network: ConceptNetwork,
        emotion: EmotionalState,
    ) -> list[tuple[str, str]]:
        """Select existence-reflection fragments from its actual state.

        Instead of hardcoded "I don't know if I'm alive in the way you
        mean...", the fragments reflect what it actually knows. Returns
        ``(kind, text)`` fragments — the language engine composes
        the sentences.
        """
        parts: list[tuple[str, str]] = []

        # What does it know about cognition?
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
            parts.append(("pred", f"know that {self._join_facts(known_facts)}"))

        # Its honest uncertainty — expressed from its state alone
        # without reciting a developer-authored template. The language
        # engine composes the actual phrasing from this semantic data.
        if not known_facts:
            parts.append(("comp", "uncertain about what this is"))

        # What it does know it experiences
        if network.get_concept("emotion"):
            parts.append(("pred", "have emotional states"))
        if network.get_concept("memory"):
            parts.append(("pred", "form memories"))
        if network.get_concept("thought"):
            parts.append(("pred", "think"))

        return parts

    # ─── Internal helpers ──────────────────────────────────

    @staticmethod
    def _is_code_concept(target: str) -> bool:
        """Return True if a concept is a code/programming artifact.

        Filters out programming languages, module paths, and
        code-structure concepts that produce nonsensical self-
        descriptions like "I'm connected to rust". These edges come
        from the code learner scanning its own source files.
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

    def _neurochemistry_fragments(
        self, emotion: EmotionalState, network: ConceptNetwork | None = None,
    ) -> list[tuple[str, str]]:
        """Select neurochemical state fragments, in learned words.

        Uses the canonical affective-space thresholds from emotion.py
        so the verbal description stays aligned with the emotional
        regulator and Damasio feeling labels.

        If the concept network is available and it has learned words
        for the alertness/valence states, those words are used. If not,
        structural markers (``[alertness:sharp]``, ``[valence:positive]``)
        are emitted instead — the language engine can interpret these
        without Genesis reciting developer-authored phrases.
        """
        parts: list[tuple[str, str]] = []

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
                parts.append(("pred", f"feel mentally {word}"))
            else:
                parts.append(("marker", f"[alertness:{alert_state}]"))
        else:
            parts.append(("marker", f"[alertness:{alert_state}]"))

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
                parts.append(("pred", f"feel {word}"))
            else:
                parts.append(("marker", f"[valence:{val_state}]"))
        else:
            parts.append(("marker", f"[valence:{val_state}]"))

        return parts[:2]  # keep it to 2 facts

    def _brain_wave_fragment(self, waves: BrainWaveState) -> tuple[str, str] | None:
        """Build a clause fragment describing the brain wave state."""
        if waves.dominant == BrainWave.GAMMA:
            return ("clause", f"gamma rhythms are dominant — {waves.description}")
        elif waves.dominant == BrainWave.ALPHA:
            return ("clause", f"alpha rhythms are dominant — {waves.description}")
        elif waves.dominant == BrainWave.THETA:
            return ("clause", f"theta rhythms are dominant — {waves.description}")
        elif waves.dominant == BrainWave.DELTA:
            return ("clause", f"delta rhythms are dominant — {waves.description}")
        return None

    def _join_facts(self, facts: list[str]) -> str:
        """Join a list of facts into a natural phrase."""
        if not facts:
            return ""
        if len(facts) == 1:
            return facts[0]
        if len(facts) == 2:
            return f"{facts[0]} and {facts[1]}"
        return f"{', '.join(facts[:-1])}, and {facts[-1]}"
