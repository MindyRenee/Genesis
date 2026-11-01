"""TemporalLobe — comprehension, lexicon, and semantic grounding.

Wernicke's area (comprehension), the anterior temporal semantic hub
(grounding, topic resolution), and the medial-temporal encoding
machinery (concept learning, episodic store) live here. The
network-only machinery is grown at construction; the pieces wired to
prefrontal state (synapses, memory store) grow as projections arrive —
the frontal lobe calls the grow_* methods when it has the deps, but
the objects live on this region.
"""

from __future__ import annotations

from typing import Any

from temporal_lobe.comprehension import Comprehension
from temporal_lobe.concept_learner import ConceptLearner
from temporal_lobe.language_latent_space import LanguageLatentSpace
from temporal_lobe.memory_store import MemoryStore
from temporal_lobe.semantic_grounding import SemanticGrounder
from temporal_lobe.topic_resolver import TopicResolver


class TemporalLobe:
    """The temporal region: heard words become meaning here.

    Attributes:
        comprehension: Wernicke's mechanism — text to propositions.
        semantic_grounder: binds surface forms to concepts (ATL hub).
        topic_resolver: extracts the discourse topic from input.
        concept_learner: word-label acquisition (grown with synapses).
        memory_store: episodic encoding (grown with frontal deps).
    """

    def __init__(self, network) -> None:
        self.network = network
        # Wernicke — utterance comprehension
        self.comprehension = Comprehension()
        # Shared geometry for the same parsed language used by grounding.
        self.language_latent_space = LanguageLatentSpace()
        # Anterior temporal semantic hub — grounding + topics
        self.semantic_grounder = SemanticGrounder(network)
        self.topic_resolver = TopicResolver(network)
        # Medial temporal — grown as projections arrive
        self.concept_learner: ConceptLearner | None = None
        self.memory_store: MemoryStore | None = None
        # The hippocampal-entorhinal organ — grid phases, place nodes,
        # anchors, transition graph. It is domain-agnostic: the world
        # channel feeds it grid coordinates, and the general channel
        # feeds it state-space position (see prefrontal). Orientation
        # is a general function, not a game function.
        from temporal_lobe.cognitive_map import CognitiveMap

        self.cognitive_map = CognitiveMap()

    def observe_language(self, text: str, result: Any) -> dict[str, Any]:
        # Update language geometry from a real comprehension result.
        return self.language_latent_space.observe(text, result)

    def grow_concept_learner(self, synapses) -> ConceptLearner:
        """Grow the label-acquisition machinery on the synaptic substrate."""
        self.concept_learner = ConceptLearner(network=self.network, synapses=synapses)
        return self.concept_learner

    def grow_memory_store(self, **deps: Any) -> MemoryStore:
        """Grow episodic encoding from the deps the caller projects in."""
        self.memory_store = MemoryStore(**deps)
        return self.memory_store
