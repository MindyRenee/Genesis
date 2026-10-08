"""Temporal subsystem — memory, recognition, and auditory processing.

════════════════════════════════════════════════════════════════════════
ANATOMY AND FUNCTION
════════════════════════════════════════════════════════════════════════

The temporal subsystem is the brain's center for memory, object recognition,
and auditory processing. It contains the hippocampal system (memory
formation and retrieval), the inferotemporal cortex (object
recognition — the end of the ventral "what" stream), and the auditory
cortex (sound processing).

Key functions:
    - Episodic memory formation and retrieval (hippocampus)
    - Object recognition and categorization (inferotemporal cortex)
    - Auditory processing (superior temporal gyrus)
    - Semantic memory consolidation (neocortical storage)
    - Pattern separation and completion (dentate gyrus, CA3)
    - Spatial memory (entorhinal cortex, place/grid cells)
    - Emotional memory (amygdala-temporal interactions)
    - Language comprehension (Wernicke's area, left temporal)

The temporal subsystem receives input from:
    - Occipital subsystem (ventral stream: V1 -> V4 -> VTC -> IT)
    - Auditory nerve (cochlea -> brainstem -> thalamus -> A1)
    - Parietal subsystem (spatial context for object-location binding)
    - Frontal subsystem (top-down attention and retrieval cues)

Pipeline (ventral "what" stream, continued from occipital subsystem):

    VTC output (from occipital subsystem)
        |
        v
    [IT] Inferotemporal cortex (object_recognition.py)
        |   Object categorization and identification
        |   YOLOv5s approximates IT object-selective neurons
        |   Produces: DetectedObject list (name, confidence, position)
        v
    [MTL bridge] (in vision — VTC -> concept embeddings)
        |   Visual-concept association
        v
    [Hippocampus] (memory/ package)
        |   Episodic memory formation and retrieval
        |   Pattern separation (dentate gyrus)
        |   Pattern completion (CA3 attractor)
        |   Systems consolidation (hippocampal -> neocortical)
        v
    [Neocortex] (concept_network + memory/semantic.py)
        |   Semantic memory consolidation
        |   Schema formation

Pipeline (auditory "what" stream):

The auditory "what" stream:

    cochlea -> brainstem -> thalamus (MGN)
        |
        v
    [A1] Primary auditory cortex (auditory.py)
        |   Spectral feature extraction
        |   Sound event classification (speech/music/impact/noise/nature)
        |   Spectrotemporal receptive fields
        v
    [Belt/Parabelt] Higher auditory areas
        |   Phonological processing, sound categorization
        v
    [Wernicke's area] Language comprehension (left temporal)


════════════════════════════════════════════════════════════════════════
MATHEMATICAL FOUNDATION
════════════════════════════════════════════════════════════════════════

Hippocampal Memory Systems
---------------------------

CA3 Recurrent Attractor (pattern completion):

The CA3 region has extensive recurrent collaterals that form an
autoassociative network. Memories are stored as attractors via
Hebbian learning.

    Storage:  W_ij = sum_mu  xi_i^mu * xi_j^mu
    Retrieval: r_i = phi( sum_j W_ij * r_j + I_i )

    where:
        W_ij    = recurrent synaptic weight
        xi^mu   = memory pattern mu (binary vector)
        I_i     = external input (partial cue)
        phi     = activation function (typically sigmoid)
        r_i     = firing rate of neuron i

A partial cue I_i activates the nearest stored attractor via pattern
completion. The attractor dynamics pull the activity toward the
stored pattern even if the cue is noisy or incomplete.

In Genesis: memory/systems.py AttractorNetwork implements this.
Patterns are stored via Hebbian outer products and retrieved via
iterative dynamics. The auto_associate() method performs pattern
completion.

Dentate Gyrus Pattern Separation:

The dentate gyrus orthogonalizes similar inputs to distinct sparse
codes, preventing interference between similar memories.

    minimize  ||x - Phi * z||^2  +  lambda * ||z||_1
    subject to  z >= 0

    where:
        x       = entorhinal cortex input
        Phi     = expansion matrix (sparse, high-dimensional)
        z       = sparse dentate code
        lambda  = sparsity weight

The expansion from EC to dentate (much larger) combined with sparse
coding ensures similar inputs map to orthogonal representations.

In Genesis: the attractor network's similarity threshold (0.75)
serves a pattern-separation function — similar patterns above
threshold are treated as the same, below threshold as distinct. A
proper dentate model would add explicit sparse expansion.

Grid Cells (medial entorhinal cortex):

Grid cells fire at hexagonal lattices of locations, providing a
metric for space. They use a continuous attractor with path
integration.

    tau * dr/dt = -r + W(r) + v(t) . grad(r)

    where:
        W(r)       = recurrent connectivity (Mexican-hat on torus)
        v(t)       = animal velocity
        grad(r)    = spatial gradient of activity
        r          = firing rate field

The velocity-driven shift moves the activity bump according to
self-motion, producing hexagonal firing patterns. Multiple modules
with geometrically spaced grid scales (lambda_n = lambda_0 * r^n,
r ~ 1.7) provide a multi-scale spatial code.

In Genesis: not yet implemented. Would be added when spatial
memory and navigation are needed.

Place Cells (hippocampus):

Place cells fire at single locations, derived from grid cell inputs
via sparse coding.

    minimize  ||x_grid - Phi_place * z||^2  +  lambda * ||z||_1

    where:
        x_grid    = vector of grid cell activities
        Phi_place = place-cell dictionary
        z         = sparse place code (typically 1 active cell)

In Genesis: not yet implemented.

Laplace Transform Temporal Representation (Howard et al., 2014):

Time is represented via leaky integrators at multiple timescales.

    F(s, t) = integral_0^t  f(tau) * e^{-s(t-tau)} dtau

    where:
        f(tau) = input function (events)
        s      = timescale parameter (inverse time constant)
        F(s,t) = Laplace transform at timescale s, time t

The inverse Laplace transform recovers temporal history. By
integrating velocity, the same framework codes spatial position.

In Genesis: the memory engine's decay and consolidation dynamics
approximate this — memories decay at multiple timescales (STM
ring buffer, LTM store, spaced repetition scheduler).


Auditory Cortex (Spectrotemporal Receptive Fields)
----------------------------------------------------

The primary auditory cortex (A1) analyzes sound via
spectrotemporal receptive fields (STRFs) — kernels that capture
the time-frequency structure of natural sounds.

    r(t) = integral integral  h(tau, f) * s(t - tau, f) dtau df

    where:
        h(tau, f) = STRF kernel (time-frequency receptive field)
        s(t, f)   = spectrogram (sound energy at time t, freq f)
        r(t)      = neural response at time t

A1 encodes fine acoustic details (frequency, timing). Higher
auditory areas (belt, parabelt) encode abstracted phonological
features and sound categories.

In Genesis: auditory.py implements a simplified version using
spectral features (RMS energy, spectral centroid, zero-crossing
rate, spectral flux, rolloff) computed via FFT on 1024-sample
blocks. Classification is rule-based (not learned STRFs) —
speech/music/impact/noise/nature categories are determined by
threshold heuristics on the spectral features. This is
functionally equivalent to A1's role (sound categorization) but
not biologically detailed.


Object Recognition (Inferotemporal Cortex)
--------------------------------------------

The inferotemporal (IT) cortex contains neurons selective for
complex shapes and object categories. It's the final stage of the
ventral "what" pathway.

In primates, IT neurons show invariance to position, scale, and
viewpoint — they respond to "faces" or "chairs" regardless of
where they appear. This invariance is built up through the
hierarchy: V1 -> V2 -> V4 -> PIT -> CIT -> AIT.

In Genesis: object_recognition.py uses YOLOv5s (a deep CNN) to
approximate IT function. YOLOv5s is trained on COCO (80 object
categories) and produces bounding boxes with class scores. It is
not biologically accurate but serves the same functional role:
turning pixels into object labels with position invariance.

The mathematical model (if we were to build one):

    features = CNN(image)                    # hierarchical features
    class_scores = W * features + b         # classification head
    bbox = regressor(features)              # bounding box regression

    class = argmax(class_scores)
    confidence = softmax(class_scores)[class]

In practice, YOLOv5s handles all of this internally. We treat it
as a black-box approximation of IT cortex.


Dream Synthesis (structurally-similar far-pair proposal + waking validation)
-----------------------------------------------------------------------------

During REM (high acetylcholine, low norepinephrine) the synthesizer
goes beyond random association: it finds concept pairs that are
*structurally similar but far apart* in the graph — same relation
signature, distant positions — and proposes novel typed edges
between them as hypotheticals at low confidence:

    hypothetical edge weight = 0.15

On the next wake (or a dream-review pass), each hypothetical is
validated by abductive/analogical inference over independent
evidence. An edge supported by >= 2 corroborating paths is promoted:

    promoted edge weight = 0.45

Unsupported hypotheticals decay. This is the sleep-time generation
of genuinely new structure — the same far-transfer mechanism
analogical reasoning uses while awake, run unconstrained.

In Genesis: sleep/dream_synthesis.py DreamSynthesisEngine implements
the propose/validate cycle; DreamProposal, DreamValidationResult,
and DreamInsight are its records.


Sleep Compression (bounded memory, honestly modeled)
------------------------------------------------------

The compressor keeps Genesis bounded: VQ codebook retraining
(re-quantizing the embedding space), edge migration into the
holographic graph, and episode summarization with raw-episode
deletion past the LTM cap.

Modeling note: this is *memory compaction*, inspired by but not
equivalent to biological consolidation. Biological systems
consolidation is selective replay and systems-level transformation
in which hippocampal episodic traces are gradually integrated into
neocortical semantic networks (Rasch & Born, 2013) — it is not
destructive deletion. The compact-LTM phase here deletes raw
episodes after summarization; that is an engineering bound, not a
claim about biology.

In Genesis: sleep/compression.py SleepCompressor implements the
pass (VQ iters, salience threshold, homeostasis scale, and merge
threshold are its tuning surface).


STDP (timing-dependent synaptic efficacy)
-------------------------------------------

STDP is a timing-dependent learning rule over a synaptic substrate
deliberately separate from the semantic graph: ConceptNetwork edges
encode explicit semantic relationships, embeddings encode
representational similarity, and synaptic efficacy encodes learned
directed strength between active units. For Δt = t_post − t_pre:

    Δt > 0:  Δw = +A⁺ · exp(-Δt / τ⁺)     (LTP — pre before post)
    Δt < 0:  Δw = −A⁻ · exp(Δt / τ⁻)      (LTD — post before pre)

    τ⁺ = τ⁻ = 20 ms, A⁺ = 0.01, A⁻ = 0.012, window = 40 ms

The slight A⁻ > A⁺ bias is homeostatic — unpaired activity tends
weakly toward depression rather than runaway potentiation. The
timing rule is followed by a three-factor neuromodulatory gate:
eligibility alone does not write; the neuromodulatory context
decides whether the timed pairing consolidates.

In Genesis: learning/stdp.py STDP over learning/synapses.py
SynapticStore. HebbianPlasticity (above, claimed from learning/) is
the correlation-based counterpart that adapts the embedding space;
STDP is the timing-based rule that adapts the efficacy substrate.


════════════════════════════════════════════════════════════════════════
BRAIN WAVES
════════════════════════════════════════════════════════════════════════

The temporal subsystem is dominated by theta oscillations (hippocampal
memory) and gamma (active processing), with important roles for
alpha and beta.

Theta (4-8 Hz) — The hippocampal memory rhythm
-------------------------------------------------

Theta is the dominant rhythm of the hippocampus during active
exploration and memory formation. Theta-gamma coupling is the
mechanism by which memories are encoded: gamma cycles nest within
theta cycles, and the theta phase determines whether encoding or
retrieval occurs (Hasselmo et al., 2002).

    Encoding: gamma on theta rising phase
    Retrieval: gamma on theta falling phase

In Genesis: the brain wave system tracks theta as the memory
rhythm. The memory engine's consolidation dynamics (sleep
replay, systems consolidation) are theta-gated. The
compute_theta_gamma_coupling() function in brain_waves.py
measures this coupling.

Gamma (30-100 Hz) — Active processing and binding
----------------------------------------------------

Gamma in the temporal subsystem reflects active memory processing —
the binding of features into coherent memory representations.
Hippocampal gamma is nested within theta (Buzsáki, 2006).

In Genesis: gamma is tracked globally by the brain wave system.
The temporal subsystem's memory retrieval and recognition processes
contribute to gamma power through their activation dynamics.

Alpha (8-12 Hz) — Cortical idling and gating
----------------------------------------------

Temporal alpha reflects the idling state of temporal cortex. Low
alpha (desynchronization) during active memory retrieval and
language processing; high alpha during idle states.

In Genesis: alpha is tracked globally. The temporal subsystem's
contribution to alpha is through its processing state — active
recognition and memory retrieval desynchronize alpha.

Beta (13-30 Hz) — Top-down control
------------------------------------

Beta in the temporal subsystem reflects top-down control from the
frontal subsystem — goal-directed retrieval, selective attention to
specific memories or objects. Beta coordinates the
frontotemporal network during memory retrieval.

In Genesis: beta is tracked globally. The temporal subsystem's
interaction with the frontal subsystem during goal-directed memory
retrieval contributes to beta-band activity.


════════════════════════════════════════════════════════════════════════
ANATOMICAL BOUNDARIES
════════════════════════════════════════════════════════════════════════

The temporal subsystem is bounded by:
    - Occipital subsystem (behind): receives ventral stream output
      (V1 -> V4 -> VTC -> IT)
    - Parietal subsystem (above): receives spatial context for
      object-location binding
    - Frontal subsystem (in front): receives retrieval cues, sends
      top-down attention
    - Brainstem (below): receives auditory nerve input

The memory/ package is a multi-subsystem system:
    - Hippocampal memory (memory/engine.py, memory/systems.py) is
      temporal subsystem
    - Working memory (memory/working.py) is frontal subsystem (PFC)
    - Procedural memory (memory/procedural.py) is basal ganglia
    - Semantic memory (memory/semantic.py) is neocortical
      (temporal-parietal)
    - Spaced repetition (memory/spaced_repetition.py) is a
      learning mechanism, not subsystem-specific

Because memory/ spans multiple subsystems, it stays at the top level
and is referenced by this subsystem, the frontal subsystem, and the basal
ganglia. This subsystem documents the hippocampal components as
temporal subsystem functions.

The MTL bridge (temporal_lobe/mtl_bridge.py, projected through
occipital_lobe) is anatomically temporal but
functionally part of the ventral visual stream. It's documented
in both places — the occipital subsystem describes its role in the
visual pipeline, and this subsystem describes its anatomical location.


════════════════════════════════════════════════════════════════════════
MODULE ORGANIZATION
════════════════════════════════════════════════════════════════════════

Modules in this folder (purely temporal subsystem):

    auditory.py
        AuditoryCortex — spectral feature extraction, sound event
        classification (speech/music/impact/noise/nature). Runs in
        a background thread, captures audio via sounddevice, computes
        FFT-based features on 1024-sample blocks.

    object_recognition.py
        ObjectRecognizer — YOLOv5s-based object detection and
        identification. Approximates inferotemporal (IT) cortex
        function: turning pixels into object labels with position
        invariance. Detects 80 COCO categories.

Top-level modules referenced by this subsystem (multi-subsystem):

    (top-level) memory/
        MemoryEngine — episodic memory retrieval and context
        (hippocampal, temporal subsystem)
        AttractorNetwork — CA3-like pattern completion
        (hippocampal, temporal subsystem)
        EmotionalMemorySystem — emotional memory tagging
        (amygdala-temporal)
        PrimingSystem — priming/association activation
        (temporal subsystem)
        SpreadingActivation — spreading activation across network
        (temporal subsystem)
        WorkingMemory — Baddeley working memory model
        (frontal subsystem — referenced by control)
        ProceduralMemory — skills and habits
        (basal ganglia — claimed by basal_ganglia)
        SemanticMemory — semantic knowledge store
        (neocortical, temporal-parietal)
        SpacedRepetitionScheduler — spaced repetition
        (learning mechanism, not subsystem-specific)

    (top-level) language/
        The language engine — Genesis's speech system. Wernicke's
        area (comprehension.py) and Broca's area (generator.py,
        graph_walk.py, grammar.py). Language acquisition
        (acquisition.py), figurative language (figurative.py),
        vocabulary (vocabulary.py), voice input/output (voice.py).
        Anatomically the language network spans left temporal
        (Wernicke's) and left frontal (Broca's), but the temporal
        subsystem is the primary semantic center.

    (concepts/) network.py
        ConceptNetwork — the semantic graph. Nodes are concepts,
        edges are relationships. This is the anterior temporal
        subsystem's semantic hub — the region that binds multimodal
        features into unified concept representations. Used by
        nearly every module.

    (concepts/) embeddings.py
        EmbeddingStore — the latent semantic space. Provides
        sub-symbolic vector representations for concepts. This
        is the temporal subsystem's latent semantic space beneath the
        symbolic concept network.

    (top-level) wordnet_dictionary.py
        WordNetDictionary — Genesis's primary word reference. Looks
        up definitions, synonyms, hypernyms. Anatomically the
        temporal subsystem stores lexical-semantic knowledge.

    (top-level) vq_codebook.py
        VQCodebook — vector quantization of the embedding space.
        Compresses the 210-dimensional embeddings into discrete
        codes. Temporal subsystem semantic compression.

    (concepts/) holographic.py
        HolographicAssociativeGraph — fixed-size associative memory.
        Replaces auto-generated bridge edges with a holographic
        representation. Temporal-limbic associative memory.

    (concepts/) archive.py
        ConceptArchive — long-term concept storage. Two-tier
        memory mirroring biological memory: active (hippocampal)
        and archived (neocortical). Temporal subsystem memory tiers.

    (concepts/) edge_proposer.py
        EdgeProposer — discovers new concept relationships from
        embedding proximity. Runs during sleep (hippocampal replay
        + consolidation). Temporal subsystem + sleep interaction.

    (concepts/) topology.py
        NetworkTopology — graph-theoretic metrics on the concept
        network. Computes clustering, small-world properties,
        hub detection. Temporal subsystem knowledge structure analysis.

    (top-level) ambient.py
        AmbientListener — speech recognition via Vosk. Captures
        audio and transcribes utterances. The auditory-temporal
        pathway: cochlea -> auditory cortex -> Wernicke's area.

    speech.py
        Voice — text-to-speech (Piper) and speech-to-text (Vosk).
        Motor output of language (Broca's area + motor cortex).
        Anatomically frontal but functionally part of the language
        network centered in the temporal subsystem.

    (top-level) brain_waves.py
        Brain wave system — tracks theta, gamma, alpha, beta, delta
        across all subsystems. Referenced by every subsystem.

    mtl_bridge.py
        MemoryBridge — the VTC -> concept embedding projection.
        Owned here: the medial temporal lobe is the ventral
        "what" stream's endpoint. ``occipital_lobe`` imports it
        across the boundary rather than duplicating it.

Top-level modules newly claimed by this subsystem:

    (cognition/) concept_learner.py, topic_resolver.py, memory_store.py
        ConceptLearner — word labeling and activation spreading into
        the network (semantic learning). TopicResolver — topic
        extraction and concept variant matching (lexical-semantic
        lookup). MemoryStore — conversation and cognitive event
        persistence (episodic recording).

    (sleep package) dream_synthesis.py, compression.py
        DreamSynthesisEngine — REM-time proposal of novel typed edges
        between structurally similar far-apart concepts, validated on
        waking. SleepCompressor — the bounded-memory compaction pass
        (VQ retraining, holographic edge migration, episode
        summarization). Both are hippocampal-replay processes riding
        on the sleep state the brainstem sets.

    (concepts/) consolidation.py, dynamics.py, extraction.py,
                archival.py, edge_log.py, types.py
        The semantic hub's machinery: ConsolidationMixin (sleep
        processing and review bridges), DynamicsMixin (spreading
        activation and cortical ticks), ExtractionMixin (concept
        extraction from text), ArchivalMixin (the archival tier),
        EdgeLog (the append-only edge journal with web provenance),
        and the CorticalLayer/Provenance types.

    (perception/) core.py, recognition.py
        Intent / QuestionType — input-side comprehension
        classification (what kind of speech act, what kind of
        question) — Wernicke's-side parsing. FaceRecognizer —
        YuNet detection + SFace 128-d embeddings; the fusiform face
        area (occipitotemporal; also re-exported by occipital_lobe).

    (language/) figurative.py, statistical_learner.py, grounding.py,
                base.py, morphology.py
        FigurativeLanguageProcessor — metaphor (cross-domain "X is
        Y") and irony detection. StatisticalLanguageLearner — n-gram
        patterns and user-style adaptation from heard input.
        SemanticGrounder — resolves heard names against the network
        without mutating it. base.py — the Thought /
        SyntacticStructure / LanguageEngine interface both language
        halves implement. morphology.py — the shared inflectional
        engine: comprehension deconjugates ("runs" → "run"),
        production conjugates ("they run" vs "it runs"); one module
        so the two systems never disagree about English.

    (learning/) stdp.py, synapses.py
        STDP over SynapticStore — timing-dependent synaptic efficacy,
        deliberately separate from the semantic graph and embeddings
        (see the STDP section above). The timing-based counterpart to
        HebbianPlasticity's correlation-based embedding adaptation.

    (self/) learning.py
        SelfDirectedLearner — deriving new network facts from what it
        already knows (reasoning, inference, conversation) instead of
        fetching from external sources. Semantic learning, temporal.

    (sleep package) architecture.py — the consolidation machinery
        SynapticDownscaler (the synaptic homeostasis downscaler —
        SHY's slow-wave downscaling) and HippocampalReplay
        (ReplayItem — the replay that drives consolidation and dream
        content). The ultradian SleepCycleTracker from the same
        module is claimed by brainstem.
"""

from __future__ import annotations

from .._views import view_getattr

# Internal modules — the auditory cortex (A1) and inferotemporal
# object recognition (the ventral "what" stream's endpoint). These
# are genuinely temporal-subsystem code, not cross-subsystem re-exports, so
# they stay eager — vision.py's `from .temporal_lobe import
# DetectedObject` resolves them directly.
from .auditory import AuditoryCortex as AuditoryCortex
from .auditory import SoundEvent as SoundEvent
from .mtl_bridge import MemoryBridge as MemoryBridge
from .mtl_bridge import TrainingExample as TrainingExample
from .object_recognition import (
    COCO_CLASSES as COCO_CLASSES,
)
from .object_recognition import (
    DetectedObject as DetectedObject,
)
from .object_recognition import (
    ObjectRecognizer as ObjectRecognizer,
)
from .vtc import VTCFeatureSpace as VTCFeatureSpace

# The temporal subsystem's multi-subsystem modules stay at the top level and
# are lazily re-exported here so the anatomy is a real connection
# layer (the association pattern). Lazy resolution is required:
# the language and perception packages are upstream in the import
# graph, so eagerly importing them here creates cycles.
_EXPORTS: dict[str, str] = {
    # Hippocampal system — memory formation, retrieval, consolidation
    "AttractorNetwork": "memory",
    "EmotionalMemorySystem": "memory",
    "Fact": "memory",
    "MemoryContext": "memory",
    "MemoryEngine": "memory",
    "MemoryRecord": "memory",
    "PrimingSystem": "memory",
    "ReconsolidationModification": "memory",
    "ReviewRecord": "memory",
    "Schema": "memory",
    "SemanticMemory": "memory",
    "SpacedRepetitionScheduler": "memory",
    "SpreadingActivation": "memory",
    # Hippocampus -> neocortex consolidation (complementary learning
    # systems) and hippocampal Hebbian plasticity
    "DualSystemLearner": "learning",
    "HebbianPlasticity": "learning",
    "HippocampalEpisode": "learning",
    "NeocorticalMemory": "learning",
    # Wernicke's area — language comprehension and semantic
    # vocabulary. Production (Broca's) is re-exported by control.
    "ComprehensionEngine": "language",
    "ComprehensionResult": "language",
    "LanguageAcquisition": "language",
    "Vocabulary": "language",
    # Anterior temporal semantic hub — the concept network binds
    # multimodal features into unified concept representations
    "Concept": "concepts",
    "ConceptCategory": "concepts",
    "ConceptModality": "concepts",
    "ConceptNetwork": "concepts",
    "Edge": "concepts",
    "RelationType": "concepts",
    "detect_category": "concepts",
    "detect_modality": "concepts",
    # The latent semantic space beneath the symbolic network
    "EmbeddingStore": "concepts",
    # Lexical-semantic knowledge — the dictionary
    "WordNetEntry": "infrastructure.wordnet_dictionary",
    "lookup_definition": "infrastructure.wordnet_dictionary",
    "lookup_word": "infrastructure.wordnet_dictionary",
    # Semantic compression and associative memory
    "HolographicGraph": "concepts",
    "VQCodebook": "infrastructure.vq_codebook",
    # Long-term concept storage — active (hippocampal) and archived
    # (neocortical) tiers
    "ConceptArchive": "concepts",
    "open_archive": "concepts",
    # Hippocampal-replay-driven edge discovery during sleep, and
    # graph-theoretic analysis of the knowledge structure
    "EdgeProposer": "concepts",
    "NetworkTopology": "concepts",
    # The auditory-temporal pathway: cochlea -> A1 -> Wernicke's area
    "AmbientListener": "perception.ambient",
    # Speech motor interface — anatomically frontal (Broca's + motor
    # cortex) but functionally part of the temporal-centered language
    # network
    "Voice": "temporal_lobe.speech",
    "VoiceInput": "temporal_lobe.speech",
    # Conversational semantic learning and variant matching —
    # word labeling into the network, topic extraction
    "ConceptLearner": "cognition.concept_learner",
    "MemoryStore": "cognition.memory_store",
    "TopicResolver": "cognition.topic_resolver",
    # Hippocampal replay content — dream synthesis (structurally
    # similar far-pair edge proposals, validated on waking) and the
    # sleep compression pass (VQ retraining, holographic edge
    # migration, episode summarization)
    "DreamInsight": "sleep.dream_synthesis",
    "DreamProposal": "sleep.dream_synthesis",
    "DreamSynthesisEngine": "sleep.dream_synthesis",
    "SleepCompressor": "sleep.compression",
    # The concept network's machinery — consolidation, activation
    # dynamics, extraction, archival tiers, the edge log, and the
    # layer/provenance types (all part of the ATL semantic hub)
    "ArchivalMixin": "concepts.archival",
    "ConsolidationMixin": "concepts.consolidation",
    "CorticalLayer": "concepts.types",
    "DynamicsMixin": "concepts.dynamics",
    "EdgeLog": "concepts.edge_log",
    "ExtractionMixin": "concepts.extraction",
    "open_edge_log": "concepts.edge_log",
    "Provenance": "concepts.types",
    # Input-side comprehension classification — intent and question
    # type detection over heard language (Wernicke's-side parsing)
    "Intent": "perception.core",
    "QuestionType": "perception.core",
    # Fusiform face area — occipitotemporal; the ventral stream's
    # people endpoint (also re-exported by occipital_lobe)
    "DetectedFace": "perception.recognition",
    "FaceRecognizer": "perception.recognition",
    "KnownFace": "perception.recognition",
    # Figurative language — metaphor and irony interpretation
    # (cross-domain semantic structure)
    "FigurativeLanguageProcessor": "language.figurative",
    "IronyDetection": "language.figurative",
    "Metaphor": "language.figurative",
    # Statistical language learning — n-gram patterns and user-style
    # adaptation from heard input
    "NGramModel": "language.statistical_learner",
    "StatisticalLanguageLearner": "language.statistical_learner",
    # Semantic grounding — resolving heard names against the network
    # without mutating it
    "GroundedProposition": "language.grounding",
    "SemanticGrounder": "language.grounding",
    # The language interface — Thought, SyntacticStructure, and the
    # LanguageEngine base both production and comprehension implement
    "LanguageEngine": "language.base",
    "SyntacticStructure": "language.base",
    "Thought": "language.base",
    # Lexical morphosyntax — the shared inflectional engine both
    # comprehension (deconjugation) and production (conjugation) use
    "conjugate_verb": "language.morphology",
    "deconjugate_verb": "language.morphology",
    "is_verb_form": "language.morphology",
    # The synaptic substrate and its timing-dependent rule —
    # experience-dependent efficacy kept deliberately separate from
    # the semantic graph (HebbianPlasticity, above, is the
    # correlation-based counterpart)
    "SpikeEvent": "learning.stdp",
    "STDP": "learning.stdp",
    "SynapticStore": "learning.synapses",
    # Self-directed knowledge learning — deriving new network facts
    # from what it already knows (reasoning, inference, conversation)
    "InferenceResult": "self.learning",
    "LearningEvent": "self.learning",
    "SelfDirectedLearner": "self.learning",
    # Sleep architecture's consolidation machinery — the synaptic
    # homeostasis downscaler (SHY) and hippocampal replay (the
    # ultradian tracker is claimed by brainstem)
    "HippocampalReplay": "sleep.architecture",
    "ReplayItem": "sleep.architecture",
    "SynapticDownscaler": "sleep.architecture",
    # Concept-network utilities — word-sense stripping and the
    # world-concept test used by the learners
    "is_world_concept": "concepts",
    "strip_sense_suffix": "concepts",
}

# The five eager names stay in a literal list so static checkers
# (pyflakes) recognize them as intentional re-exports.
__all__ = [
    "COCO_CLASSES",
    "AuditoryCortex",
    "DetectedObject",
    "MemoryBridge",
    "ObjectRecognizer",
    "SoundEvent",
    "TrainingExample",
    "VTCFeatureSpace",
]
__all__ += sorted(_EXPORTS)

__getattr__ = view_getattr(_EXPORTS, __name__)
