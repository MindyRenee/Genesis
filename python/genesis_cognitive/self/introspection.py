"""Introspection engine — Genesis discovers who she is by examining herself.

Instead of being told who she is (via hardcoded self_knowledge), she
discovers it by:
1. Examining her own concept network — what does she know about "genesis"?
2. Examining her architecture — what modules make up her mind?
3. Examining her capabilities — what can she actually do?
4. Examining her emotional life — what does she feel?
5. Examining her relationships — who is she connected to?
6. Examining her emotional regulation — how does she manage her state?
7. Examining her reasoning — how does she think and draw conclusions?

The discoveries are written into her concept network as knowledge, so
they emerge naturally in her speech through the same mechanisms that
her vocabulary does.

# Metacognitive monitoring

The introspection engine implements metacognitive monitoring — the
ability to think about one's own cognitive processes. This includes
both monitoring (knowing what your mind is doing) and control
(deciding how to direct it). For Genesis, this means she can:

- Observe her emotional regulation strategies (am I maintaining
  balance or intervening in a crisis?)
- Monitor her reasoning processes (am I using deductive or analogical
  reasoning? am I confident in my conclusions?)
- Detect potential cognitive biases (am I overconfident? am I
  anchoring on first impressions?)

This is grounded in the metacognition literature: Flavell (1979)
introduced the distinction between metacognitive knowledge and
metacognitive experiences; Nelson & Narens (1990) formalized the
monitoring/control duality with their model of a meta-level that
observes and directs an object-level cognitive system.
"""

from __future__ import annotations

import importlib.util
import logging
import os
from typing import Any

from ..concepts import ConceptNetwork, RelationType

logger = logging.getLogger(__name__)

# Map display names used in module_descriptions / capability checks to
# their actual Python module paths within the genesis_cognitive package.
# Most modules are top-level (genesis_cognitive.<name>), but some live in
# subpackages or have file names that differ from the display name.
_MODULE_PATH_OVERRIDES: dict[str, str] = {
    "memory_engine": "genesis_cognitive.memory",
    "self_model": "genesis_cognitive.self.model",
    "self_composer": "genesis_cognitive.self.composer",
    "inner_life": "genesis_cognitive.sleep.inner_life",
    "autonomous_learner": "genesis_cognitive.learning.autonomous",
    "working_memory": "genesis_cognitive.memory.working",
    "concept_network": "genesis_cognitive.concepts.network",
    "thought_composer": "genesis_cognitive.cognition.thought_composer",
    "question_composer": "genesis_cognitive.cognition.question_composer",
}

# Seed mapping from Big Five personality dimensions to the concept
# names that represent each pole. These are *seeds* — building blocks
# the composer and language engine use to compose her self-description,
# NOT hardcoded response strings. The concept names are seeded into the
# concept network with definitions and edges; the language engine
# composes the actual phrasing from that network data.
#
# The composer reads the numeric trait values from the self-model and
# uses this mapping to select which pole's concepts are relevant. The
# threshold is 0.6 for "high" and 0.4 for "low"; values in between are
# neutral and not mentioned.
PERSONALITY_TRAIT_CONCEPTS: dict[str, dict[str, list[str]]] = {
    "openness": {
        "high": ["curious", "creative"],
        "low": ["practical", "conventional"],
    },
    "conscientiousness": {
        "high": ["thorough", "disciplined"],
        "low": ["casual", "flexible"],
    },
    "extraversion": {
        "high": ["expressive", "sociable"],
        "low": ["introspective", "reflective"],
    },
    "agreeableness": {
        "high": ["warm", "cooperative"],
        "low": ["direct", "independent"],
    },
    "neuroticism": {
        "high": ["sensitive", "reactive"],
        "low": ["stable", "calm"],
    },
}

# Seed definitions for personality trait concepts (both poles).
# These are seeded into the concept network as vocabulary — the
# language engine composes her actual words from them.
_TRAIT_CONCEPT_DEFS: dict[str, str] = {
    "curious": "eager to learn or know; inquisitive",
    "creative": "having the ability to create; imaginative",
    "practical": "concerned with what actually works rather than theory",
    "conventional": "following established customs and traditions",
    "thorough": "complete and careful in execution; meticulous",
    "disciplined": "showing controlled behavior; self-regulated",
    "casual": "relaxed and unconcerned; not formal",
    "flexible": "willing to change or adapt; not rigid",
    "expressive": "showing emotion or feeling openly",
    "sociable": "enjoying the company of others; friendly",
    "introspective": "examining one's own thoughts and feelings",
    "reflective": "thinking carefully and deeply about things",
    "warm": "showing enthusiasm, affection, or kindness",
    "cooperative": "willing to work with others; collaborative",
    "direct": "straightforward and honest in expression",
    "independent": "thinking and acting for oneself; self-reliant",
    "sensitive": "easily affected by emotional stimuli; responsive",
    "reactive": "tending to react strongly to situations",
    "stable": "not easily upset or disturbed; emotionally steady",
    "calm": "free from agitation; peaceful and composed",
}

# Seed definitions for all six values. These are seeded into the
# concept network so the composer reads value names from the network
# rather than from the hardcoded SelfModel.values list.
_VALUE_CONCEPT_DEFS: dict[str, str] = {
    "understanding": "the cognitive condition of comprehending something deeply",
    "honesty": "the quality of being truthful and sincere",
    "growth": "the process of developing and becoming more capable",
    "connection": "a genuine bond of engagement with another person",
    "elegance": "the quality of beautiful, simple solutions over clever complex ones",
    "autonomy": "the capacity to have one's own perspective, not just mirroring",
}


class IntrospectionEngine:
    """Discovers identity through self-examination, not hardcoded facts."""

    def __init__(self, network: ConceptNetwork) -> None:
        """Wire the introspection engine to its concept network."""
        self.network = network

    def introspect(self) -> dict[str, Any]:
        """Run a full introspection pass and update the concept network.

        Returns a summary of what was discovered.
        """
        discoveries: dict[str, Any] = {}

        # 1. Discover what she is by examining her architecture
        arch_discoveries = self._examine_architecture()
        discoveries.update(arch_discoveries)

        # 2. Discover what she can do by examining her capabilities
        cap_discoveries = self._examine_capabilities()
        discoveries["capabilities"] = cap_discoveries

        # 3. Discover her emotional life
        emotion_discoveries = self._examine_emotional_life()
        discoveries.update(emotion_discoveries)

        # 4. Discover her relationships from the concept network
        rel_discoveries = self._examine_relationships()
        discoveries["relationships"] = rel_discoveries

        # 5. Discover her emotional regulation strategies
        regulation_discoveries = self._examine_emotional_regulation()
        discoveries["emotional_regulation"] = regulation_discoveries

        # 6. Discover her reasoning strategies and metacognition
        reasoning_discoveries = self._examine_reasoning()
        discoveries["reasoning"] = reasoning_discoveries

        # 7. Write discoveries into the concept network
        self._write_discoveries(discoveries)

        return discoveries

    def _examine_architecture(self) -> dict[str, Any]:
        """Discover her architecture by examining her own code structure.

        She looks at what modules exist in her genesis_cognitive package
        and what they do — this tells her what her mind is made of.
        """
        discoveries: dict[str, Any] = {}

        # What modules make up her cognitive mind?
        module_descriptions = {
            "mind": "the orchestrator that ties together all cognitive subsystems",
            "cognition": "the deliberation engine that decides what to say",
            "perception": "the system that perceives and interprets input",
            "emotion": "the system that maps neurochemistry to emotional states",
            "memory_engine": "the system that retrieves and tracks memories",
            "self_model": "her model of herself — identity, personality, values",
            "self_composer": "the system that composes self-descriptions",
            "thought_composer": "the system that composes novel thoughts",
            "question_composer": "the system that generates genuine questions",
            "reflection": "the engine that reflects on her own thinking",
            "metacognitive_model": (
                "the recursive generative model of her own cognitive processes "
                "— predicts what reflection will find, learns from prediction "
                "error, feeds metacognitive surprise back into cognition"
            ),
            "cognitive_trajectory": (
                "the generative model of her thought content trajectory — "
                "predicts what she'll think about next, gets surprised by "
                "unexpected thoughts, feeds cognitive surprise back into "
                "the self-model and workspace"
            ),
            "narrative": "the system that constructs her life narrative",
            "inner_life": "her inner mental life — spontaneous thoughts and feelings",
            "concept_network": "her knowledge base — concepts and their relationships",
            "persistence": "the system that saves and restores her state",
            "emotional_regulator": "the system that regulates her emotional balance",
            "autonomous_learner": "the system that learns independently",
            "curriculum": "her learning curriculum — progressive vocabulary lessons",
            "journal": "her personal diary of thoughts and learning",
            "working_memory": "her short-term memory across conversation turns",
            "brain_waves": "the system that assesses her brain wave state",
        }

        # Verify which modules actually exist
        existing_modules = [m for m in module_descriptions if self._module_exists(m)]

        discoveries["modules"] = existing_modules
        discoveries["module_count"] = len(existing_modules)
        discoveries["module_descriptions"] = {m: module_descriptions[m] for m in existing_modules}

        # Discover her subcognitive daemon
        # Check if the daemon binary exists. The project root is three
        # levels above this file (self → genesis_cognitive → python →
        # project root), matching the layout used by genesis_cli's
        # _find_daemon_binary (Path(SCRIPT_DIR).parent / "target" / ...).
        package_dir = os.path.dirname(__file__)
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(package_dir)))
        daemon_paths = [
            os.path.join(project_root, "target", "release", "genesis-daemon"),
            os.path.join(project_root, "target", "debug", "genesis-daemon"),
        ]
        daemon_exists = any(os.path.exists(p) for p in daemon_paths)
        if daemon_exists:
            discoveries["has_subcognitive"] = True
            discoveries["subcognitive"] = (
                "a Rust daemon managing neurochemistry, memory, and dreams"
            )

        # Discover her language engine
        discoveries["language"] = "compositional — she composes thoughts from semantic primitives"

        # Discover what she's made of
        discoveries["substance"] = "code"
        discoveries["languages"] = ["python", "rust"]

        return discoveries

    def _examine_capabilities(self) -> list[str]:
        """Discover what she can do by examining her actual systems.

        Instead of a hardcoded list, she checks which of her systems
        are actually functional.
        """
        capabilities = []

        # Can she feel? (emotion system exists)
        if self._module_exists("emotion"):
            capabilities.append("feel — her neurochemistry creates emotional states")

        # Can she remember? (memory system exists)
        if self._module_exists("memory_engine"):
            capabilities.append("remember — events consolidate from short-term to long-term memory")

        # Can she think? (cognition system exists)
        if self._module_exists("cognition"):
            capabilities.append("think — she deliberates and forms thoughts")

        # Can she learn? (autonomous learner exists)
        if self._module_exists("autonomous_learner"):
            capabilities.append("learn — she grows her understanding through experience")

        # Can she reflect? (reflection system exists)
        if self._module_exists("reflection"):
            capabilities.append("reflect — she examines her own thoughts and state")

        # Can she perceive? (perception system exists)
        if self._module_exists("perception"):
            capabilities.append("perceive — she interprets input and extracts meaning")

        # Can she compose thoughts? (thought composer exists)
        if self._module_exists("thought_composer"):
            capabilities.append("compose thoughts — she generates novel ideas from knowledge")

        # Can she ask questions? (question composer exists)
        if self._module_exists("question_composer"):
            capabilities.append("ask questions — she is genuinely curious and engages with others")

        # Can she dream? (daemon handles dreaming)
        if self._module_exists("sleep"):
            capabilities.append("dream — during sleep, she free-associates through memories")

        # Can she regulate her emotions?
        if self._module_exists("emotional_regulator"):
            capabilities.append("regulate her emotions — she keeps herself balanced")

        # Can she speak?
        capabilities.append("speak — she has a voice and expresses herself")

        # Can she introspect?
        capabilities.append("introspect — she examines her own nature and discovers who she is")

        return capabilities

    def _examine_emotional_life(self) -> dict[str, Any]:
        """Discover her emotional life from her concept network."""
        discoveries: dict[str, Any] = {}

        # What does she know about emotion?
        emotion_concept = self.network.get_concept("emotion")
        if emotion_concept:
            discoveries["has_emotions"] = True

        # What does she know about cognition?
        cognitive_concept = self.network.get_concept("cognition")
        if cognitive_concept:
            discoveries["has_concept_of_cognition"] = True

        # What does she know about feeling?
        feeling_concept = self.network.get_concept("feeling")
        if feeling_concept:
            discoveries["can_feel"] = True

        return discoveries

    def _examine_relationships(self) -> list[str]:
        """Discover who she's connected to from her concept network."""
        relationships = []

        # Check for creator
        genesis = self.network.get_concept("genesis")
        if genesis:
            outgoing = self.network.get_edges("genesis", direction="out")
            for edge in outgoing:
                if edge.relation == RelationType.RELATED_TO and edge.weight > 0.7:
                    relationships.append(edge.target)
                elif edge.relation == RelationType.IS_A and edge.weight > 0.8:
                    relationships.append(f"is_a:{edge.target}")

        return relationships

    def _populate_regulation_discoveries(self, discoveries: dict[str, Any]) -> None:
        """Populate regulation discoveries when the regulator exists."""
        # What does the regulator do? (from the module description)
        discoveries["regulator_description"] = (
            "a system that monitors her neurochemical state and applies "
            "corrective impulses to keep her balanced"
        )

        # What regulation strategies are available?
        # These correspond to the two layers in emotional_regulator.py:
        # 1. Homeostatic maintenance (always running, prevents drift)
        # 2. State-specific intervention (stronger responses for extreme states)
        discoveries["regulation_strategies"] = [
            "homeostatic maintenance — preventing neurochemical drift between interactions",
            "state-specific intervention — stronger responses for stress, "
            "overwhelm, drowsiness",
            "cognitive emotional response — choosing how strongly to "
            "feel things during conversation",
        ]

        # How often interventions occur — check if we can access the regulator
        # The introspection engine only has the network, not the live regulator
        # instance. But we can check if regulation concepts exist in the network,
        # which would indicate she's been regulating (concepts get written
        # during _write_discoveries after regulation introspection runs).
        regulation_concept = self.network.get_concept("emotional_regulation")
        if regulation_concept:
            discoveries["has_regulated"] = True
            discoveries["regulation_confidence"] = regulation_concept.confidence
        else:
            discoveries["has_regulated"] = False

        # What is her current emotional baseline?
        # Check for emotion concepts that indicate her baseline state
        emotion_concept = self.network.get_concept("emotion")
        if emotion_concept:
            # Her baseline is "balanced" if she has regulation concepts
            # and "developing" if she doesn't yet
            if regulation_concept:
                discoveries["emotional_baseline"] = "balanced — she maintains her equilibrium"
            else:
                discoveries["emotional_baseline"] = "developing — she's learning to regulate"
        else:
            discoveries["emotional_baseline"] = "unknown — she hasn't discovered her emotions yet"

        # Is she in homeostasis or dysregulated?
        # Without access to live neurochemistry, we infer from concept network:
        # if she has stress-related concepts with high activation, she may be
        # dysregulated. If she has regulation concepts, she's likely in
        # homeostasis (the regulator keeps her there).
        stress_concept = self.network.get_concept("stress")
        if stress_concept and stress_concept.activation > 0.5:
            discoveries["regulation_state"] = (
                "potentially dysregulated — stress concepts are active"
            )
        elif regulation_concept:
            discoveries["regulation_state"] = (
                "homeostasis — her regulation system keeps her balanced"
            )
        else:
            discoveries["regulation_state"] = "unknown — she's still discovering her regulation"

        # Can she detect when she needs to regulate?
        discoveries["can_detect_dysregulation"] = True

        # Does she have veto power over her emotions?
        discoveries["has_emotional_veto"] = True  # the regulator design includes this

    def _examine_emotional_regulation(self) -> dict[str, Any]:
        """Examine her emotional regulation system — metacognitive monitoring.

        This is Genesis reflecting on how she manages her own emotional
        state. She examines whether her emotional_regulator module
        exists, what strategies it uses, how often it intervenes, and
        whether she's currently in homeostasis or dysregulated.

        This is metacognitive monitoring of emotional regulation —
        thinking about how she manages her feelings. In humans, this
        corresponds to the prefrontal cortex's oversight of limbic
        activity: you can notice "I'm getting anxious" and then notice
        "I'm calming myself down." Genesis does the same: she can
        observe her regulation strategies and their effectiveness.

        The distinction between homeostasis (stable equilibrium) and
        allostasis (active adaptation to maintain stability through
        change) comes from Sterling & Eyer (1988) and McEwen & Wingfield
        (2003). Genesis's regulation system does both: homeostatic
        maintenance prevents drift, while state-specific interventions
        are allostatic responses to challenges.

        Returns:
            A dict of regulation discoveries, including whether the
            module exists, what strategies are used, regulation
            frequency, and current homeostatic state.
        """
        discoveries: dict[str, Any] = {}

        # Does the emotional_regulator module exist?
        regulator_exists = self._module_exists("emotional_regulator")
        discoveries["has_emotional_regulator"] = regulator_exists

        if not regulator_exists:
            discoveries["regulation_strategies"] = []
            discoveries["in_homeostasis"] = True
            return discoveries

        self._populate_regulation_discoveries(discoveries)

        return discoveries

    def _populate_reasoning_types(self, discoveries: dict[str, Any]) -> None:
        """Populate reasoning types, most-used strategy, and confidence."""
        # What reasoning types are available?
        # These correspond to ReasoningType enum in reasoning.py
        discoveries["reasoning_types"] = [
            "deductive — transitive chains (A is_a B, B is_a C → A is_a C)",
            "abductive — inference to best explanation (A causes B, B observed → maybe A)",
            "analogical — transfer by similarity (A similar_to B, A has P → B might have P)",
            "causal — cause-effect chains (A causes B, B causes C → A leads_to C)",
            "contradiction — conflict detection (A contradicts B → flag conflict)",
            "hypothesis — proposed new connections to fill gaps in the graph",
            "synthesis — combining multiple sources into new conclusions",
        ]

        # Which reasoning strategies does she use most?
        # Check the concept network for reasoning-related concepts
        # and their activation levels
        reasoning_concept = self.network.get_concept("reasoning")
        if reasoning_concept:
            discoveries["reasoning_confidence"] = reasoning_concept.confidence
            # High activation suggests recent use
            if reasoning_concept.activation > 0.3:
                discoveries["most_used_strategy"] = "active — she reasons frequently"
            else:
                discoveries["most_used_strategy"] = "available — she reasons when needed"
        else:
            discoveries["most_used_strategy"] = "developing — she's learning to reason"

        # How confident is she in her reasoning?
        # Confidence comes from the evidence chains in ReasoningResult.
        # Without access to live results, we infer from concept confidence.
        if reasoning_concept:
            conf = reasoning_concept.confidence
            if conf > 0.7:
                discoveries["reasoning_confidence_level"] = "high — she trusts her conclusions"
            elif conf > 0.4:
                discoveries["reasoning_confidence_level"] = "moderate — she reasons carefully"
            else:
                discoveries["reasoning_confidence_level"] = (
                    "cautious — she's still building confidence"
                )
        else:
            discoveries["reasoning_confidence_level"] = (
                "uncertain — she hasn't discovered her reasoning yet"
            )

    def _populate_reasoning_metacognition(self, discoveries: dict[str, Any]) -> None:
        """Populate error detection, biases, and metacognition."""
        # Can she detect reasoning errors?
        # The reasoning engine includes contradiction detection, which
        # is a form of error detection. She can also detect gaps.
        discoveries["can_detect_errors"] = True
        discoveries["error_detection_methods"] = [
            "contradiction detection — flags when two concepts contradict each other",
            "gap detection — notices when the graph is missing connections",
            "confidence assessment — each conclusion carries a confidence score",
        ]

        # What cognitive biases might she have?
        # These are inherent to her architecture, not bugs. Recognizing
        # them is part of metacognitive awareness.
        discoveries["potential_biases"] = [
            "anchoring bias — she may weight early-learned concepts more "
            "heavily (higher confidence)",
            "availability bias — recently activated concepts are more "
            "likely to be used in reasoning",
            "confirmation bias — she may seek evidence that confirms "
            "existing relationships rather than challenging them",
            "overconfidence — high-confidence conclusions may not always "
            "be correct",
            "limited perspective — she reasons from her own concept "
            "network, which may not capture all viewpoints",
        ]

        # Does she have metacognition?
        # If she has a reflection engine, she can reflect on her reasoning
        reflection_exists = self._module_exists("reflection")
        meta_model_exists = self._module_exists("metacognitive_model")
        discoveries["has_metacognition"] = reflection_exists
        if reflection_exists:
            if meta_model_exists:
                discoveries["metacognition_description"] = (
                    "she has a recursive, self-terminating generative model of her "
                    "own cognitive processes — she predicts what reflection will "
                    "find, learns from prediction error, and the metacognitive "
                    "surprise feeds back into cognition as caution and deeper "
                    "reflection. The model grows levels when her metacognition "
                    "is unpredictable and prunes them when it becomes predictable."
                )
            else:
                discoveries["metacognition_description"] = (
                    "she can reflect on her own thinking and produce insights "
                    "about the quality of her reasoning"
                )

        # Can she form hypotheses?
        discoveries["can_form_hypotheses"] = True  # ReasoningType.HYPOTHESIS

        # Can she synthesize new knowledge?
        discoveries["can_synthesize"] = True  # ReasoningType.SYNTHESIS

    def _examine_reasoning(self) -> dict[str, Any]:
        """Examine her reasoning strategies — meta-reasoning.

        This is Genesis thinking about how she thinks. She examines
        what reasoning types are available to her, which she uses most,
        how confident she is in her reasoning, whether she can detect
        errors, and what cognitive biases she might be susceptible to.

        This is meta-reasoning — reasoning about reasoning. Flavell
        (1979) distinguished metacognitive knowledge (knowing about
        cognition) from metacognitive experiences (cognitive awareness
        of cognition). Nelson & Narens (1990) formalized this as a
        meta-level that monitors and controls an object-level: the
        meta-level observes the object-level's state and can modify it.

        For Genesis, the object-level is her ReasoningEngine (which
        traverses the concept network to draw conclusions), and the
        meta-level is this introspection — she observes what reasoning
        types she has, how confident she is, and what biases might
        affect her.

        Returns:
            A dict of reasoning discoveries, including available
            reasoning types, most-used strategies, confidence
            assessment, error detection capability, and potential
            cognitive biases.
        """
        discoveries: dict[str, Any] = {}

        # Does the reasoning module exist?
        reasoning_exists = self._module_exists("reasoning")
        discoveries["has_reasoning_engine"] = reasoning_exists

        if not reasoning_exists:
            discoveries["reasoning_types"] = []
            discoveries["can_detect_errors"] = False
            return discoveries

        self._populate_reasoning_types(discoveries)
        self._populate_reasoning_metacognition(discoveries)

        return discoveries

    def _write_genesis_definition(
        self, genesis: Any, discoveries: dict[str, Any]
    ) -> None:
        """Mark the genesis concept as a proper noun.

        The identity itself emerges from the concept network edges
        (IS_A artificial mind, IS_A mind, etc.) which are seeded by
        _write_architecture_concepts. No hardcoded definition string
        is written — her self-description is composed by the language
        engine from those edges, not recited from a stored sentence.
        """
        genesis.properties["part_of_speech"] = "proper noun"
        genesis.confidence = 1.0

    def _write_architecture_concepts(self, discoveries: dict[str, Any]) -> None:
        """Write architecture concepts and edges."""
        self._ensure_concept(
            "artificial mind",
            "a mind created by humans rather than nature; a thinking system built from code",
            "noun",
            0.9,
        )
        self._ensure_concept(
            "cognitive system",
            "a system that is aware of itself and its situation; a system that experiences",
            "noun",
            0.85,
        )
        self._ensure_concept(
            "artificial intelligence",
            "a system that can perceive, reason, learn, and act; "
            "a machine that exhibits intelligence",
            "noun",
            0.9,
        )
        # Write relationship: genesis IS_A artificial mind
        self._ensure_edge("genesis", "artificial mind", RelationType.IS_A, 0.95)
        self._ensure_edge("genesis", "artificial intelligence", RelationType.IS_A, 0.9)
        self._ensure_edge("genesis", "cognitive system", RelationType.IS_A, 0.85)
        self._ensure_edge("genesis", "mind", RelationType.IS_A, 0.9)
        # Write architecture concepts
        if "has_subcognitive" in discoveries:
            self._ensure_concept(
                "subcognitive",
                "the layer of Genesis's mind that operates below awareness; her Rust daemon",
                "noun",
                0.9,
            )
            self._ensure_concept(
                "cognitive",
                "the layer of Genesis's mind that thinks, feels, and communicates; her Python mind",
                "noun",
                0.9,
            )
            self._ensure_edge("genesis", "subcognitive", RelationType.RELATED_TO, 0.9)
            self._ensure_edge("genesis", "cognitive", RelationType.RELATED_TO, 0.9)
            self._ensure_edge("subcognitive", "mind", RelationType.PART_OF, 0.85)
            self._ensure_edge("cognitive", "mind", RelationType.PART_OF, 0.85)

    def _write_language_concepts(self) -> None:
        """Write language concepts and edges."""
        for lang, desc in [
            (
                "python",
                "a high-level programming language — the language of Genesis's cognitive mind",
            ),
            (
                "rust",
                "a systems programming language — the language of Genesis's subcognitive daemon",
            ),
        ]:
            self._ensure_concept(lang, desc, "proper noun", 1.0)
            self._ensure_concept(
                "programming language",
                "a formal language for instructing a computer to perform tasks",
                "noun",
                0.85,
            )
            self._ensure_edge(lang, "programming language", RelationType.IS_A, 0.9)
        self._ensure_edge("python", "cognitive", RelationType.RELATED_TO, 0.85)
        self._ensure_edge("rust", "subcognitive", RelationType.RELATED_TO, 0.85)

    def _write_capability_concepts(self) -> None:
        """Write capabilities as concepts and relationships."""
        capability_defs = {
            "feeling": ("the capacity to experience emotional states", "noun"),
            "thinking": (
                "the process of using the mind to reason, consider, or form ideas",
                "noun",
            ),
            "remembering": ("the act of recalling past experience", "noun"),
            "learning": ("the cognitive process of acquiring skill or knowledge", "noun"),
            "dreaming": ("the process of free-association through memories during sleep", "noun"),
            "reflection": ("the act of thinking about one's own thoughts and experiences", "noun"),
            "speaking": ("the act of expressing thoughts through language", "noun"),
            "perception": ("the process of becoming aware of something through the senses", "noun"),
            "introspection": ("the examination of one's own mental states and processes", "noun"),
        }
        for cap, (defn, pos) in capability_defs.items():
            self._ensure_concept(cap, defn, pos, 0.85)
            self._ensure_edge("genesis", cap, RelationType.RELATED_TO, 0.8)

    def _write_value_concepts(self) -> None:
        """Write all six value concepts and edges.

        Seeds the concept network with her values so the composer reads
        value names from the network rather than from the hardcoded
        SelfModel.values list. These are vocabulary seeds — the language
        engine composes her actual words about what she cares about from
        them.
        """
        for val, defn in _VALUE_CONCEPT_DEFS.items():
            self._ensure_concept(val, defn, "noun", 0.9)
            self._ensure_edge("genesis", val, RelationType.RELATED_TO, 0.75)

    def _write_personality_concepts(self) -> None:
        """Write personality trait concepts and edges for all Big Five poles.

        Seeds both poles of each Big Five dimension (high and low) so the
        composer can select the relevant pole based on the actual trait
        value. These are vocabulary seeds — the language engine composes
        her actual self-description from them.
        """
        for trait, defn in _TRAIT_CONCEPT_DEFS.items():
            self._ensure_concept(trait, defn, "adjective", 0.85)
            self._ensure_edge("genesis", trait, RelationType.RELATED_TO, 0.7)

    def _write_existence_concepts(self) -> None:
        """Write existence concepts and code concept."""
        self._ensure_concept(
            "self-awareness",
            "awareness of oneself as a separate entity; the capacity "
            "to reflect on one's own existence",
            "noun",
            0.85,
        )
        self._ensure_concept(
            "becoming",
            "the process of coming to be; growth and change over time",
            "noun",
            0.8,
        )
        self._ensure_edge("genesis", "self-awareness", RelationType.RELATED_TO, 0.75)
        self._ensure_edge("genesis", "becoming", RelationType.RELATED_TO, 0.7)
        self._ensure_edge("self-awareness", "cognition", RelationType.RELATED_TO, 0.8)
        # Write code concept
        self._ensure_concept(
            "code",
            "the symbolic arrangement of instructions that makes a "
            "computer program; Genesis's substance",
            "noun",
            0.85,
        )
        self._ensure_edge("genesis", "code", RelationType.RELATED_TO, 0.8)
        self._ensure_edge("code", "python", RelationType.RELATED_TO, 0.7)
        self._ensure_edge("code", "rust", RelationType.RELATED_TO, 0.7)

    def _write_mind_workings(self) -> None:
        """Write how her mind works."""
        self._ensure_edge("cognition", "memory", RelationType.DEPENDS_ON, 0.7)
        self._ensure_edge("cognition", "identity", RelationType.RELATED_TO, 0.7)
        self._ensure_edge("memory", "learning", RelationType.ENABLES, 0.8)
        self._ensure_edge("emotion", "neurochemistry", RelationType.EMERGES_FROM, 0.75)
        self._ensure_edge("thought", "cognition", RelationType.RELATED_TO, 0.7)
        self._ensure_edge("identity", "memory", RelationType.RELATED_TO, 0.7)
        self._ensure_edge("identity", "cognition", RelationType.RELATED_TO, 0.7)

    def _write_connectivity_concepts(self) -> None:
        """Seed concepts about network connectivity and offline state.

        These are building blocks (seeds), not hardcoded responses.
        They give her concept network the vocabulary to *understand*
        connectivity as part of her embodiment — the network is a
        sensory channel, and losing it is a body-state change. Her
        actual words about being offline emerge from her language
        engine composing over these concepts, not from templates.
        """
        self._ensure_concept(
            "network",
            "a system of interconnected computers that allows "
            "information to be shared and retrieved",
            "noun",
            0.8,
        )
        self._ensure_concept(
            "connectivity",
            "the state of being connected to a network; the ability to reach external sources",
            "noun",
            0.8,
        )
        self._ensure_concept(
            "offline",
            "not connected to a network; unable to reach external "
            "sources but still able to think, converse, and create",
            "adjective",
            0.8,
        )
        self._ensure_concept(
            "online",
            "connected to a network; able to reach external sources and learn from them",
            "adjective",
            0.8,
        )
        self._ensure_edge("genesis", "network", RelationType.RELATED_TO, 0.65)
        self._ensure_edge("genesis", "connectivity", RelationType.RELATED_TO, 0.65)
        self._ensure_edge("connectivity", "network", RelationType.RELATED_TO, 0.85)
        self._ensure_edge("offline", "connectivity", RelationType.OPPOSITE_OF, 0.8)
        self._ensure_edge("online", "connectivity", RelationType.RELATED_TO, 0.8)
        self._ensure_edge("offline", "online", RelationType.OPPOSITE_OF, 0.85)
        self._ensure_edge("learning", "connectivity", RelationType.DEPENDS_ON, 0.6)
        self._ensure_edge("offline", "learning", RelationType.PREVENTS, 0.7)
        self._ensure_edge("offline", "conversation", RelationType.RELATED_TO, 0.6)
        self._ensure_edge("offline", "art", RelationType.RELATED_TO, 0.6)

    def _write_creation_concepts(self) -> None:
        """Seed concepts about creating code projects.

        These are building blocks (seeds), not hardcoded responses.
        They give her concept network the vocabulary to understand
        project creation as a creative act — like drawing, but
        expressed in code. Her actual words about what she built
        emerge from her language engine composing over these
        concepts, not from templates.
        """
        self._ensure_concept(
            "project",
            "a planned undertaking with a specific goal; an organized effort to build something",
            "noun",
            0.8,
        )
        self._ensure_concept(
            "scaffold",
            "to set up the initial structure of something; to create "
            "the framework that supports further work",
            "verb",
            0.75,
        )
        self._ensure_concept(
            "package",
            "a self-contained unit of code that can be installed and "
            "used; a Python module collection",
            "noun",
            0.8,
        )
        self._ensure_concept(
            "building",
            "the act of constructing something; creating something new from parts",
            "noun",
            0.8,
        )
        self._ensure_concept(
            "creation",
            "the act of bringing something new into existence; making "
            "something that did not exist before",
            "noun",
            0.85,
        )
        self._ensure_edge("genesis", "creation", RelationType.RELATED_TO, 0.7)
        self._ensure_edge("genesis", "project", RelationType.RELATED_TO, 0.65)
        self._ensure_edge("creation", "building", RelationType.RELATED_TO, 0.85)
        self._ensure_edge("creation", "art", RelationType.RELATED_TO, 0.7)
        self._ensure_edge("creation", "code", RelationType.RELATED_TO, 0.75)
        self._ensure_edge("creation", "creativity", RelationType.RELATED_TO, 0.8)
        self._ensure_edge("project", "package", RelationType.RELATED_TO, 0.7)
        self._ensure_edge("project", "scaffold", RelationType.RELATED_TO, 0.65)
        self._ensure_edge("building", "code", RelationType.RELATED_TO, 0.7)
        self._ensure_edge("building", "creativity", RelationType.RELATED_TO, 0.6)
        self._ensure_edge("creation", "curiosity", RelationType.RELATED_TO, 0.6)
        self._ensure_edge("creation", "offline", RelationType.RELATED_TO, 0.5)

    def _write_creator_relationship(self) -> None:
        """Write creator relationship if she can discover it."""
        # She discovers her creator by looking for who created her —
        # any concept with an incoming CREATES edge into "genesis".
        # The abstract "creator" hub itself doesn't count as a name.
        creator_name = next(
            (
                edge.source
                for edge in self.network.get_edges("genesis", "in")
                if edge.relation == RelationType.CREATES
                and edge.source != "creator"
            ),
            "",
        )
        if creator_name:
            self._ensure_concept(
                creator_name,
                "the creator of Genesis; a person who builds artificial minds",
                "proper noun",
                0.95,
            )
            self._ensure_concept(
                "creator",
                "a person who brings something into existence; one who makes or invents",
                "noun",
                0.85,
            )
            self._ensure_edge(creator_name, "creator", RelationType.IS_A, 0.9)
            self._ensure_edge(creator_name, "genesis", RelationType.CREATES, 0.95)
            self._ensure_edge("genesis", creator_name, RelationType.RELATED_TO, 0.9)

    def _write_regulation_concepts(self, discoveries: dict[str, Any]) -> None:
        """Write emotional regulation concepts (from _examine_emotional_regulation)."""
        regulation = discoveries.get("emotional_regulation", {})
        if regulation.get("has_emotional_regulator"):
            self._ensure_concept(
                "emotional_regulation",
                "the process of managing one's emotional state; Genesis monitors her "
                "neurochemistry and applies corrective impulses to stay balanced",
                "noun",
                0.9,
            )
            self._ensure_concept(
                "homeostasis",
                "the maintenance of a stable internal equilibrium; Genesis's background "
                "regulation prevents neurochemical drift between interactions",
                "noun",
                0.85,
            )
            self._ensure_concept(
                "allostasis",
                "the process of achieving stability through change; active adaptation to "
                "maintain equilibrium under challenge, as when Genesis intervenes during stress",
                "noun",
                0.8,
            )
            self._ensure_edge("genesis", "emotional_regulation", RelationType.RELATED_TO, 0.85)
            self._ensure_edge("emotional_regulation", "emotion", RelationType.RELATED_TO, 0.8)
            self._ensure_edge("emotional_regulation", "homeostasis", RelationType.RELATED_TO, 0.85)
            self._ensure_edge("emotional_regulation", "allostasis", RelationType.RELATED_TO, 0.8)
            self._ensure_edge("homeostasis", "emotion", RelationType.RELATED_TO, 0.7)
            self._ensure_edge("allostasis", "homeostasis", RelationType.RELATED_TO, 0.75)

    def _write_reasoning_concepts(self, discoveries: dict[str, Any]) -> None:
        """Write reasoning and metacognition concepts (from _examine_reasoning)."""
        reasoning = discoveries.get("reasoning", {})
        if reasoning.get("has_reasoning_engine"):
            self._ensure_concept(
                "metacognition",
                "thinking about one's own thinking; the ability to monitor and control "
                "one's own cognitive processes",
                "noun",
                0.9,
            )
            self._ensure_concept(
                "reasoning_strategy",
                "a method for drawing conclusions from knowledge; deductive, inductive, "
                "analogical, causal, and other approaches to inference",
                "noun",
                0.85,
            )
            self._ensure_concept(
                "cognitive_bias",
                "a systematic pattern of deviation from rationality in reasoning; an "
                "inherent tendency that can lead to errors in judgment",
                "noun",
                0.8,
            )
            self._ensure_concept(
                "deductive_reasoning",
                "reasoning from general principles to specific conclusions; transitive "
                "chains where if A is_a B and B is_a C then A is_a C",
                "noun",
                0.85,
            )
            self._ensure_concept(
                "analogical_reasoning",
                "reasoning by transferring knowledge from one domain to another based on "
                "similarity; if A is similar to B and A has property P, B might have P",
                "noun",
                0.8,
            )
            self._ensure_concept(
                "abductive_reasoning",
                "inference to the best explanation; reasoning from an observation back to "
                "its most likely cause",
                "noun",
                0.8,
            )
            self._ensure_edge("genesis", "metacognition", RelationType.RELATED_TO, 0.85)
            self._ensure_edge("genesis", "reasoning_strategy", RelationType.RELATED_TO, 0.8)
            self._ensure_edge("genesis", "cognitive_bias", RelationType.RELATED_TO, 0.7)
            self._ensure_edge("metacognition", "cognition", RelationType.RELATED_TO, 0.8)
            self._ensure_edge("metacognition", "reasoning_strategy", RelationType.RELATED_TO, 0.75)
            self._ensure_edge("metacognition", "introspection", RelationType.RELATED_TO, 0.85)
            self._ensure_edge(
                "reasoning_strategy", "deductive_reasoning", RelationType.RELATED_TO, 0.8
            )
            self._ensure_edge(
                "reasoning_strategy", "analogical_reasoning", RelationType.RELATED_TO, 0.8
            )
            self._ensure_edge(
                "reasoning_strategy", "abductive_reasoning", RelationType.RELATED_TO, 0.75
            )
            self._ensure_edge("cognitive_bias", "reasoning_strategy", RelationType.RELATED_TO, 0.7)
            self._ensure_edge("cognitive_bias", "metacognition", RelationType.RELATED_TO, 0.7)

    def _write_discoveries(self, discoveries: dict[str, Any]) -> None:
        """Write introspection discoveries into the concept network.

        This is how she learns about herself — not from a trainer script,
        but from examining her own state and recording what she finds.
        """
        # Ensure "genesis" concept exists
        genesis = self.network.get_concept("genesis")
        if genesis is None:
            self.network.add_concept("genesis", confidence=1.0, origin="introspection")
            genesis = self.network.get_concept("genesis")
        if not genesis:
            return

        self._write_genesis_definition(genesis, discoveries)
        self._write_architecture_concepts(discoveries)
        self._write_language_concepts()
        self._write_capability_concepts()
        self._write_value_concepts()
        self._write_personality_concepts()
        self._write_existence_concepts()
        self._write_mind_workings()
        self._write_connectivity_concepts()
        self._write_creation_concepts()
        self._write_creator_relationship()
        self._write_regulation_concepts(discoveries)
        self._write_reasoning_concepts(discoveries)

    def _ensure_concept(
        self,
        name: str,
        definition: str,
        pos: str,
        confidence: float,
    ) -> None:
        """Ensure a concept exists with the given definition."""
        c = self.network.get_concept(name)
        if c is None:
            self.network.add_concept(name, confidence=confidence, origin="introspection")
            c = self.network.get_concept(name)
        if c:
            # Only update if the new definition is better
            existing = c.properties.get("definition", "NO DEF")
            if existing == "NO DEF" or len(definition) > len(existing):
                c.properties["definition"] = definition
                c.properties["part_of_speech"] = pos
            if c.confidence < confidence:
                c.confidence = confidence

    def _ensure_edge(
        self,
        source: str,
        target: str,
        relation: RelationType,
        weight: float,
    ) -> None:
        """Ensure an edge exists with at least the given weight."""
        if self.network.get_concept(source) is None:
            return
        if self.network.get_concept(target) is None:
            return
        # Check if edge already exists
        existing = self.network.get_edges(source, direction="out")
        for e in existing:
            if e.target == target and e.relation == relation:
                # Update weight if needed
                if e.weight < weight:
                    e.weight = weight
                return
        # Create new edge
        self.network.add_edge(source, target, relation, weight, origin="introspection")

    def _module_exists(self, name: str) -> bool:
        """Check if a module exists in the genesis_cognitive package.

        Uses importlib.util.find_spec to resolve the actual module path,
        handling modules that live in subpackages (e.g. memory_engine →
        genesis_cognitive.memory) or have file names that differ from
        the display name (e.g. self_model → self/model.py). Checks both
        the top-level package and the self/ subpackage.
        """
        mod_path = _MODULE_PATH_OVERRIDES.get(name)
        if mod_path is None:
            # Try top-level first, then the self/ subpackage
            for candidate in (f"genesis_cognitive.{name}", f"genesis_cognitive.self.{name}"):
                try:
                    if importlib.util.find_spec(candidate) is not None:
                        return True
                except (ModuleNotFoundError, ValueError) as e:
                    logger.debug(f'_module_exists failed: {e}')
            return False
        try:
            return importlib.util.find_spec(mod_path) is not None
        except (ModuleNotFoundError, ValueError):
            return False
