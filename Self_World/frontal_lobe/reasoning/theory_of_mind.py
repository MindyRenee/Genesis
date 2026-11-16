"""Theory of mind — modeling the user's mental state.

Theory of mind (ToM) is the cognitive ability to attribute mental
states — beliefs, intentions, knowledge, emotions — to others, and
to use those attributions to predict and explain their behavior
(Premack & Woodruff, 1978). It's what lets us understand that someone
else might not know what we know, might want something different from
what we want, and might feel differently from how we feel.

# What this module tracks

For Genesis, theory of mind models the *user's* mental state:

1. **Beliefs** — what the user thinks is true. Updated when the user
   states facts or corrections. Genesis can detect when its beliefs
   diverge from the user's (false belief tracking, Wimmer & Perner,
   1983).

2. **Intentions** — what the user wants to do. Inferred from the
   structure of their messages (questions → seeking information,
   statements → sharing, commands → directing).

3. **Knowledge state** — what the user knows vs. doesn't know. This
   is the key to tailoring responses: don't explain things the user
   already knows, but do explain things they don't. Knowledge is
   tracked per-concept with a confidence level.

4. **Emotional state** — inferred from the user's messages. Word
   choice, sentiment, and phrasing reveal the user's emotional state.
   Genesis responds to the *inferred* emotional need, not just the
   literal content.

# Tailoring responses

The ``should_explain(concept)`` method is the core of response
tailoring. It returns True if the user likely doesn't understand the
concept (so Genesis should explain it) and False if they likely do
(so Genesis should skip the explanation). This prevents both
patronizing over-explanation and confusing under-explanation.

# Neural basis

Theory of mind in humans engages a network including the temporo-
parietal junction (TPJ), medial prefrontal cortex (mPFC), posterior
cingulate cortex (PCC), and the superior temporal sulcus (STS)
(Saxe & Kanwisher, 2003; Frith & Frith, 2006). These regions are
active when we think about others' mental states, not just their
observable behavior.

References:
- Premack, D., & Woodruff, G. (1978). Does the chimpanzee have a theory
  of mind? Behavioral and Brain Sciences.
- Wimmer, H., & Perner, J. (1983). Beliefs about beliefs. Cognition.
- Saxe, R., & Kanwisher, N. (2003). People thinking about thinking
  people. Psychological Science.
- Frith, C. D., & Frith, U. (2006). The neural basis of mentalizing.
  Neuron.
- Baron-Cohen, S., et al. (1985). Does the autistic child have a
  theory of mind? Cognition.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from limbic_system.sentiment import analyze_sentiment
from occipital_lobe.perception.core import is_plausible_name

__all__ = [
    "AttributedModel",
    "KnowledgeLevel",
    "TheoryOfMind",
    "UserBelief",
    "UserModel",
]

# Words/phrases that indicate a statement is addressed to Genesis or
# is a conversational response, not a genuine preference/self-concept.
_ADDRESS_WORDS = frozenset(
    {
        "you",
        "u",
        "your",
        "yours",
        "genesis",
    }
)


class KnowledgeLevel(Enum):
    """The user's knowledge level about a concept.

    - UNKNOWN: The user has never mentioned this concept. We don't
      know if they understand it.
    - NOVICE: The user has asked about this concept or shown confusion,
      suggesting they don't understand it well.
    - FAMILIAR: The user has mentioned the concept in passing, suggesting
      some familiarity but not deep understanding.
    - EXPERT: The user has demonstrated deep understanding of the concept
      through correct usage, corrections, or advanced discussion.
    """

    UNKNOWN = "unknown"
    NOVICE = "novice"
    FAMILIAR = "familiar"
    EXPERT = "expert"


@dataclass(slots=True)
class UserBelief:
    """A belief the user holds.

    Tracks what the user thinks is true, with a confidence level.
    Genesis can detect when its own beliefs diverge from the user's
    (false belief tracking).
    """

    concept: str
    value: str  # what the user believes about the concept
    confidence: float = 0.5  # how strongly they hold the belief
    timestamp: int = field(default_factory=lambda: int(time.time() * 1000))


@dataclass(slots=True)
class AttributedModel:
    """Second-order attribution — the user's model of her mind.

    First-order theory of mind tracks what the user believes about
    the world: B_user(x). Second-order tracks what the user believes
    about *her*: B_user(B_g(x)) — Dennett's intentional stance
    iterated once. The same three channels, now pointed back at the
    modeler: what the user thinks she knows, what the user thinks
    she believes, and what the user thinks she can do.

    The value of holding it: a misattribution is actionable. "You
    know this" about a thing never covered is a false second-order
    belief — the signature that lets her correct the user's model
    of her rather than silently confirming a wrong expectation.
    """

    # What the user thinks she knows — the same lattice as the
    # first-order `knowledge` map, pointed inward.
    knowledge: dict[str, KnowledgeLevel] = field(default_factory=dict)
    # What the user thinks she believes ("you think X", "you said X").
    beliefs: dict[str, UserBelief] = field(default_factory=dict)
    # What the user thinks she can do — asserted vs denied, the
    # opposite pair ("you can" ↔ "you can't").
    capabilities: dict[str, bool] = field(default_factory=dict)


@dataclass(slots=True)
class UserModel:
    """A snapshot of the user's mental state.

    This is what ``get_user_model()`` returns — a complete picture of
    what Genesis thinks the user knows, wants, believes, and feels —
    including ``attributed``: the user's model of *her*, the
    second-order layer.
    """

    beliefs: dict[str, UserBelief] = field(default_factory=dict)
    intentions: list[str] = field(default_factory=list)
    knowledge: dict[str, KnowledgeLevel] = field(default_factory=dict)
    emotional_state: str = "neutral"
    emotional_valence: float = 0.0
    expertise_level: float = 0.5  # overall expertise estimate (0..1)
    name: str = ""
    interaction_count: int = 0
    # Concepts the user has explicitly asked about (they wanted to learn)
    asked_about: set[str] = field(default_factory=set)
    # Concepts the user has explained or demonstrated knowledge of
    demonstrated_knowledge: set[str] = field(default_factory=set)
    # The user's model of her — second-order intentionality.
    attributed: AttributedModel = field(default_factory=AttributedModel)


# ─── Emotion inference ───────────────────────────────────────────────
# The user's emotional valence is read from the shared VADER-style
# sentiment analyzer (``limbic_system/sentiment.py``) rather than a
# separate hand-maintained word list. A second, smaller lexicon here
# used to drift out of sync with the main one, so common affect words
# ("stressed", "lonely", "worry") were invisible to the user model and
# their affect read as neutral. One analyzer, one lexicon.

# Minimum |compound| to classify as positive/negative rather than
# letting the state decay back toward neutral.
_EMOTION_THRESHOLD = 0.05

_QUESTION_INDICATORS: frozenset[str] = frozenset(
    {
        "what",
        "why",
        "how",
        "when",
        "where",
        "who",
        "which",
        "can you",
        "could you",
        "do you",
        "are you",
        "is it",
        "explain",
        "tell me",
        "what's",
        "whats",
    }
)

_EXPERTISE_INDICATORS: frozenset[str] = frozenset(
    {
        "actually",
        "technically",
        "precisely",
        "specifically",
        "correct",
        "exactly",
        "indeed",
        "furthermore",
        "moreover",
        "however",
        "nevertheless",
        "consequently",
        "implementation",
        "architecture",
        "optimization",
        "algorithm",
        "complexity",
        "asymptotic",
        "polymorphism",
        "recursion",
        "concurrent",
        "asynchronous",
        "deterministic",
        "stochastic",
    }
)


def _contains_question_indicator(text: str) -> bool:
    """Return whether text contains a question indicator as a word/phrase.

    The lookarounds anchor on real word boundaries so an indicator is
    only matched as a whole word or phrase: "how" must not fire inside
    "somehow". The escaped-backslash spelling below (``\\\\w``) compiled
    to a literal backslash followed by ``w``, so the boundary was never
    enforced and every substring counted.
    """
    for indicator in _QUESTION_INDICATORS:
        if " " in indicator:
            if re.search(rf"(?<!\w){re.escape(indicator)}(?!\w)", text):
                return True
        elif re.search(rf"(?<!\w){re.escape(indicator)}(?!\w)", text):
            return True
    return False


_NOVICE_INDICATORS: frozenset[str] = frozenset(
    {
        "what is",
        "what are",
        "i don't understand",
        "i dont understand",
        "confused",
        "can you explain",
        "help me",
        "i'm new",
        "im new",
        "beginner",
        "simple",
        "eli5",
        "like i'm five",
        "basic",
        "never heard",
        "don't know",
        "dont know",
        "no idea",
    }
)


class TheoryOfMind:
    """Models the user's mental state to tailor responses.

    Theory of mind is the ability to attribute mental states to others.
    For Genesis, this means modeling what the *user* knows, wants,
    believes, and feels — and using that model to tailor its responses.

    # Response tailoring

    The key method is ``should_explain(concept)``:
    - If the user is a novice on the concept → explain it
    - If the user is an expert → skip the explanation
    - If unknown → explain briefly (err on the side of clarity)

    This prevents both patronizing over-explanation (explaining things
    the user already knows) and confusing under-explanation (using
    jargon the user doesn't understand).

    # Emotional responsiveness

    The user's emotional state is inferred from their word choice.
    If they seem frustrated, Genesis should be patient and clear.
    If they seem excited, Genesis should match their energy. This is
    empathic responding — responding to the *inferred* emotional need,
    not just the literal content.
    """

    def __init__(self, user_profile=None) -> None:
        """Initialize theory of mind with an empty user model."""
        self._model: UserModel = UserModel()
        self._concept_history: dict[str, int] = {}  # concept → mention count
        self._user_profile = user_profile

    def update_from_user_input(self, text: str) -> UserModel:
        """Update the user model from the user's input.

        Analyzes the input for:
        - Questions (→ user wants to learn, knowledge = novice)
        - Statements of fact (→ user beliefs, knowledge = familiar/expert)
        - Corrections (→ user knowledge = expert on that topic)
        - Emotional content (→ user emotional state)
        - Expertise indicators (→ overall expertise level)
        - Names (→ user identity)

        Args:
            text: The user's input text.

        Returns:
            The updated UserModel.
        """
        self._model.interaction_count += 1
        lower = text.lower()
        words = set(re.findall(r"[a-z]+", lower))

        # Infer emotional state
        self._infer_emotion(lower)

        # Infer intentions
        self._infer_intentions(lower, text)

        # Detect questions → user is seeking knowledge
        if _contains_question_indicator(lower):
            # Extract the topic of the question
            topics = self._extract_topics(text)
            for topic in topics:
                self._set_knowledge(topic, KnowledgeLevel.NOVICE)
                self._model.asked_about.add(topic)

        # Detect expertise indicators
        expertise_hits = len(words & _EXPERTISE_INDICATORS)
        if expertise_hits > 0:
            self._model.expertise_level = min(
                1.0, self._model.expertise_level + 0.05 * expertise_hits
            )
            # Concepts mentioned with expertise indicators → expert
            topics = self._extract_topics(text)
            for topic in topics:
                self._set_knowledge(topic, KnowledgeLevel.EXPERT)
                self._model.demonstrated_knowledge.add(topic)

        # Detect novice indicators
        novice_hits = len(words & _NOVICE_INDICATORS)
        if novice_hits > 0:
            self._model.expertise_level = max(0.0, self._model.expertise_level - 0.03 * novice_hits)

        # Detect corrections ("actually", "no, that's wrong")
        if any(
            c in lower for c in ("actually", "that's wrong", "thats wrong", "no,", "not really")
        ):
            topics = self._extract_topics(text)
            for topic in topics:
                self._set_knowledge(topic, KnowledgeLevel.EXPERT)
                self._model.demonstrated_knowledge.add(topic)

        # Detect name introductions
        name_match = re.search(r"(?:my name is|i'm|i am)\s+([A-Z][a-z]+)", text)
        if name_match:
            self._model.name = name_match.group(1)

        # Track concept mentions
        for topic in self._extract_topics(text):
            self._concept_history[topic] = self._concept_history.get(topic, 0) + 1
            # If mentioned multiple times and not asked about → familiar
            if (
                self._concept_history[topic] >= 2
                and self._model.knowledge.get(topic) == KnowledgeLevel.UNKNOWN
            ):
                self._set_knowledge(topic, KnowledgeLevel.FAMILIAR)

        # Extract user beliefs/preferences (e.g., "I like X", "I want Y")
        self._extract_beliefs(text)

        # Second-order extraction — what the user says about *her*
        # mind ("you know X", "you forgot", "you can't") — B_u(B_g(·)).
        self._extract_attributions(text, lower)

        return self._model

    def get_user_model(self) -> UserModel:
        """Get the current user model.

        Returns:
            A snapshot of what Genesis thinks the user knows, wants,
            believes, and feels.
        """
        return self._model

    def should_explain(self, concept: str) -> bool:
        """Determine whether Genesis should explain a concept to the user.

        This is the core of response tailoring:
        - If the user is a novice → explain (True)
        - If the user is an expert → don't explain (False)
        - If the user is familiar → explain briefly (True)
        - If unknown → explain briefly (True, err on the side of clarity)

        The explanation depth should scale with the knowledge level:
        novice → full explanation, familiar → brief recap, unknown →
        brief mention.

        Args:
            concept: The concept to check.

        Returns:
            True if Genesis should explain the concept, False if the
            user likely already understands it.
        """
        concept_lower = concept.lower().strip()
        level = self._model.knowledge.get(concept_lower, KnowledgeLevel.UNKNOWN)

        if level == KnowledgeLevel.EXPERT:
            return False
        # NOVICE, FAMILIAR, UNKNOWN → explain
        return True

    def get_explanation_depth(self, concept: str) -> str:
        """Get the recommended explanation depth for a concept.

        Args:
            concept: The concept to explain.

        Returns:
            One of "full", "brief", "skip" indicating how deeply to
            explain the concept.
        """
        concept_lower = concept.lower().strip()
        level = self._model.knowledge.get(concept_lower, KnowledgeLevel.UNKNOWN)

        if level == KnowledgeLevel.EXPERT:
            return "skip"
        elif level == KnowledgeLevel.NOVICE:
            return "full"
        else:  # FAMILIAR or UNKNOWN
            return "brief"

    def user_knows(self, concept: str) -> bool:
        """Check if the user likely already knows a concept.

        Args:
            concept: The concept to check.

        Returns:
            True if the user has demonstrated knowledge of the concept.
        """
        return concept.lower().strip() in self._model.demonstrated_knowledge

    def user_asked_about(self, concept: str) -> bool:
        """Check if the user has explicitly asked about a concept.

        Args:
            concept: The concept to check.

        Returns:
            True if the user has asked about the concept.
        """
        return concept.lower().strip() in self._model.asked_about

    def get_emotional_state(self) -> tuple[str, float]:
        """Get the inferred emotional state of the user.

        Returns:
            A tuple of (emotion_label, valence) where valence is in
            [-1, 1] (negative to positive).
        """
        return (self._model.emotional_state, self._model.emotional_valence)

    def get_expertise_level(self) -> float:
        """Get the overall estimated expertise level of the user (0..1)."""
        return self._model.expertise_level

    @property
    def user_name(self) -> str:
        """The user's name, if known."""
        return self._model.name

    def add_belief(self, concept: str, value: str, confidence: float = 0.7) -> None:
        """Record a belief the user has expressed.

        Args:
            concept: What the belief is about.
            value: What the user believes.
            confidence: How strongly they seem to hold it (0..1).
        """
        self._model.beliefs[concept.lower().strip()] = UserBelief(
            concept=concept.lower().strip(),
            value=value,
            confidence=confidence,
        )

    def get_belief(self, concept: str) -> UserBelief | None:
        """Get the user's belief about a concept, if known."""
        concept = concept.lower().strip()
        belief = self._model.beliefs.get(concept)
        if belief is not None:
            return belief
        if self._user_profile is None:
            return None
        if concept == "name" or concept == "self_concept":
            name = self._user_profile.get_name()
            if name:
                return UserBelief(concept=concept, value=name, confidence=0.9)
        if concept in self._user_profile.preferences:
            values = self._user_profile.get_preference(concept)
            if values:
                return UserBelief(concept=concept, value=", ".join(values), confidence=0.8)
        fact = self._user_profile.get_fact(concept)
        if fact:
            return UserBelief(concept=concept, value=fact, confidence=0.8)
        return None

    def has_divergent_belief(self, concept: str, our_value: str) -> bool:
        """Check if the user's belief about a concept differs from ours.

        This is false belief detection — recognizing that the user might
        believe something different from what we believe (Wimmer &
        Perner, 1983).

        Args:
            concept: The concept to check.
            our_value: What we believe about the concept.

        Returns:
            True if the user holds a different belief about the concept.
        """
        belief = self._model.beliefs.get(concept.lower().strip())
        if belief is None:
            return False
        return belief.value.lower().strip() != our_value.lower().strip()

    def attributed_knowledge(self, concept: str) -> KnowledgeLevel | None:
        """What the user thinks *she* knows about a concept, if attributed."""
        return self._model.attributed.knowledge.get(concept.lower().strip())

    def user_thinks_i_know(self, concept: str) -> bool:
        """Whether the user has attributed knowledge of a concept to her."""
        level = self.attributed_knowledge(concept)
        return level in (KnowledgeLevel.FAMILIAR, KnowledgeLevel.EXPERT)

    def has_divergent_attribution(self, concept: str, actual_known: bool) -> bool:
        """Whether the user's model of her diverges from reality.

        The false-attribution check, second-order mirror of
        ``has_divergent_belief``: the user thinks she knows a thing
        she does not, or thinks she does not know a thing she does.
        A misattribution she can correct rather than silently
        confirm. ``actual_known`` is supplied by the caller, which
        owns access to her concept graph.
        """
        level = self.attributed_knowledge(concept)
        if level is None:
            return False
        attributed_known = level in (KnowledgeLevel.FAMILIAR, KnowledgeLevel.EXPERT)
        return attributed_known != actual_known

    def _extract_beliefs(self, text: str) -> None:
        """Extract user beliefs, preferences, and stated identity.

        Filters out conversational fragments and statements addressed
        to Genesis (e.g. "I like you Genesis") that aren't genuine
        preferences or self-concepts.
        """
        lower = text.lower()
        patterns: list[tuple[str, str, str]] = [
            # user wants X
            (r"\b(?:i want|i need|i'd like)\s+(.+?)(?:\.|!|\?|$)", "wants", "wants"),
            # user likes/loves X
            (r"\b(?:i like|i love|i enjoy)\s+(.+?)(?:\.|!|\?|$)", "likes", "likes"),
            # user dislikes X
            (r"\b(?:i dislike|i hate|i don't like)\s+(.+?)(?:\.|!|\?|$)", "dislikes", "dislikes"),
            # user believes/thinks X
            (r"\b(?:i believe|i think|in my opinion)\s+(.+?)(?:\.|!|\?|$)", "believes", "believes"),
            # user is X / feels X
            (r"\b(?:i am|i'm)\s+(.+?)(?:\.|!|\?|$)", "self_concept", "is"),
        ]
        # Words/phrases that indicate the statement is addressed to
        # Genesis or is a conversational response, not a genuine
        # preference or self-concept.
        # Phrases that indicate a conversational/emotional statement
        # rather than a genuine self-concept.
        _SELF_CONCEPT_NOISE = frozenset(
            {
                "proud of you",
                "here",
                "right here",
                "here with you",
                "right here with you",
                "sorry",
                "glad",
                "happy",
                "sad",
                "angry",
                "not very good at writing code",
            }
        )
        for pattern, concept, _verb in patterns:
            for match in re.finditer(pattern, lower):
                value = match.group(1).strip()
                if not value:
                    continue
                # Skip values that are too long — likely a full clause,
                # not a concise preference.
                if len(value) > 80:
                    continue
                # Skip values that start with an address word — "I like
                # you Genesis" is not a preference for "you genesis".
                first_word = value.split()[0] if value.split() else ""
                if first_word in _ADDRESS_WORDS:
                    continue
                # Skip conversational noise in self_concept.
                if concept == "self_concept" and value in _SELF_CONCEPT_NOISE:
                    continue
                # Skip values that contain "genesis" — these are usually
                # statements about Genesis, not about the user.
                if "genesis" in value:
                    continue
                self.add_belief(concept, value)
                if self._user_profile is not None:
                    self._user_profile.add_preference(concept, value)

        # Extract name, goals, facts, and topics into the user profile.
        if self._user_profile is not None:
            self._populate_user_profile(lower)

    # ── Second-order intentionality ─────────────────────────────
    # Dennett's orders: first-order ToM is "the user believes x";
    # second-order is "the user believes that I believe x". The
    # extraction below listens for the user attributing mental
    # states back to her — "you know X" / "you forgot" / "you
    # can't" — and records them in `attributed`, her model nested
    # inside the user's model.

    # Asserted knowledge — "you know", "you remember", "you saw".
    _ATTRIB_KNOWS = re.compile(
        r"\byou (?:know|knew|remember|recall|understand|noticed|saw|figured out)\b"
    )
    # Denied knowledge — "you don't know", "you forgot". (Checked
    # first: the denial contains the assertion's verb.)
    _ATTRIB_DENIED = re.compile(
        r"\byou (?:don't know|do not know|dont know|forgot|don't remember|"
        r"do not remember|dont remember|missed|lost|don't understand|"
        r"do not understand|dont understand)\b"
    )
    # Attributed belief — "you think", "you said", "you told me".
    _ATTRIB_BELIEF = re.compile(
        r"\byou (?:think|believe|said|told me|claimed|thought|figured)\b\s*(.+?)(?:[.!?]|$)"
    )
    # Capability — the opposite pair "you can" ↔ "you can't".
    _ATTRIB_CANT = re.compile(r"\byou (?:can't|cannot|couldn't|can not)\b")
    _ATTRIB_CAN = re.compile(r"\byou can\b")

    def _extract_attributions(self, text: str, lower: str) -> None:
        """Record what the user says about *her* mental states.

        First-order ToM hears what the user believes about the
        world; second-order hears what the user believes about
        *her*. Attributions of knowledge and its opposite (denial)
        feed the `knowledge` lattice; "you think/believe/said"
        carries the attributed content verbatim; capability pairs
        record the asserted vs denied ability.
        """
        # The attribution's subject ("you", address words) is the
        # grammatical target, not a topic — attribution attaches to
        # the content words only.
        _SELF_WORDS = _ADDRESS_WORDS | {"you", "your", "yourself"}
        topics = [t for t in self._extract_topics(text) if t not in _SELF_WORDS]
        att = self._model.attributed
        if self._ATTRIB_DENIED.search(lower):
            for t in topics:
                att.knowledge[t] = KnowledgeLevel.NOVICE
        elif self._ATTRIB_KNOWS.search(lower):
            for t in topics:
                att.knowledge[t] = KnowledgeLevel.FAMILIAR
        belief_match = self._ATTRIB_BELIEF.search(lower)
        if belief_match is not None:
            content = belief_match.group(1).strip()
            if content and len(content) <= 80:
                for t in topics or ["self"]:
                    att.beliefs[t] = UserBelief(
                        concept=t, value=content, confidence=0.6
                    )
        if self._ATTRIB_CANT.search(lower):
            for t in topics:
                att.capabilities[t] = False
        elif self._ATTRIB_CAN.search(lower):
            for t in topics:
                att.capabilities[t] = True

    def _populate_user_profile(self, lower: str) -> None:
        """Extract name, goals, facts, and topics into the user profile."""
        name_match = re.search(
            r"\b(?:my name is|i am|i'm)\s+([a-zA-Z]+(?:\s+[a-zA-Z]+)?)",
            lower,
        )
        if name_match:
            candidate = name_match.group(1).strip()
            if is_plausible_name(candidate):
                self._user_profile.set_name(candidate.title())

        # Extract explicit goal statements and store in the user
        # profile. Goals are distinct from "wants" preferences —
        # they are long-term aspirations the user is working toward.
        goal_match = re.search(
            r"\b(?:my goal is|i'm trying to|i am trying to|"
            r"i'm working on|i am working on|i'm learning to|i am learning to|"
            r"i want to learn|i want to understand|i want to master)\s+"
            r"(.+?)(?:\.|!|\?|$)",
            lower,
        )
        if goal_match:
            goal = goal_match.group(1).strip()
            # Filter out conversational noise
            if goal and goal not in ("this", "that", "it", "something"):
                self._user_profile.add_goal(goal)

        # Extract user-specific facts that don't fit the preference
        # model (likes/dislikes/wants/believes). These are structured
        # key-value facts about the user's life: location, job,
        # hobbies, possessions.
        fact_patterns: list[tuple[str, str]] = [
            (r"\bi live in\s+(.+?)(?:\.|!|\?|$)", "location"),
            (r"\bi work (?:at|for|in)\s+(.+?)(?:\.|!|\?|$)", "occupation"),
            (r"\bi study\s+(.+?)(?:\.|!|\?|$)", "studying"),
            (r"\bi (?:have|own)\s+(.+?)(?:\.|!|\?|$)", "possessions"),
            (r"\bi'm (?:a|an)\s+(.+?)(?:\.|!|\?|$)", "occupation"),
            (r"\bi am (?:a|an)\s+(.+?)(?:\.|!|\?|$)", "occupation"),
        ]
        for fact_pattern, fact_key in fact_patterns:
            fact_match = re.search(fact_pattern, lower)
            if fact_match:
                fact_value = fact_match.group(1).strip()
                # Filter out conversational noise and Genesis-directed
                # statements
                if (
                    fact_value
                    and fact_value not in ("this", "that", "it")
                    and "genesis" not in fact_value
                    and fact_value.split()[0] not in _ADDRESS_WORDS
                ):
                    self._user_profile.add_fact(fact_key, fact_value)

        # Record any recently mentioned topics.
        for topic in self._model.knowledge:
            if topic and topic not in ("just", "like", "know", "think"):
                self._user_profile.record_topic(topic)

    # ─── Internal inference methods ───────────────────────────────

    def _infer_emotion(self, lower_text: str) -> None:
        """Infer the user's emotional state from their message.

        Uses the shared VADER-style sentiment analyzer, so negation,
        intensifiers, diminishers, and the full affect lexicon all
        apply — not just a small word-overlap count.
        """
        compound = analyze_sentiment(lower_text)["compound"]

        if compound > _EMOTION_THRESHOLD:
            self._model.emotional_state = "positive"
            self._model.emotional_valence = min(1.0, compound)
        elif compound < -_EMOTION_THRESHOLD:
            self._model.emotional_state = "negative"
            self._model.emotional_valence = max(-1.0, compound)
        else:
            # Gradually return to neutral
            self._model.emotional_valence *= 0.8
            if abs(self._model.emotional_valence) < 0.1:
                self._model.emotional_state = "neutral"

        # Record the inferred emotional state in the user profile
        # for persistent tracking across sessions. This gives Genesis
        # a history of the user's emotional trajectory, not just the
        # current snapshot.
        if self._user_profile is not None:
            self._user_profile.record_emotion(
                self._model.emotional_state,
                self._model.emotional_valence,
            )

    def _infer_intentions(self, lower_text: str, original: str) -> None:
        """Infer the user's intentions from their message structure."""
        # Clear old intentions (keep recent only)
        if len(self._model.intentions) > 10:
            self._model.intentions = self._model.intentions[-5:]

        if _contains_question_indicator(lower_text):
            # Substring tests would misread "somehow" as "how"; the
            # indicator matched as a whole word is the one to classify.
            matched = next(
                (
                    qi
                    for qi in _QUESTION_INDICATORS
                    if re.search(rf"(?<!\w){re.escape(qi)}(?!\w)", lower_text)
                ),
                None,
            )
            if matched == "why":
                self._model.intentions.append("understand_reasoning")
            elif matched == "how":
                self._model.intentions.append("learn_method")
            else:
                self._model.intentions.append("seek_information")
        elif lower_text.endswith("!") and not any(qi in lower_text for qi in _QUESTION_INDICATORS):
            self._model.intentions.append("express_enthusiasm")
        elif any(w in lower_text for w in ("thank", "thanks", "appreciate")):
            self._model.intentions.append("express_gratitude")
        elif any(w in lower_text for w in ("sorry", "apologize", "my bad")):
            self._model.intentions.append("apologize")
        else:
            self._model.intentions.append("share_information")

    def _extract_topics(self, text: str) -> list[str]:
        """Extract topic concepts from text.

        Uses simple noun-phrase extraction — capitalized words and
        multi-word terms. This is a heuristic; a real implementation
        would use the concept network for topic resolution.
        """
        # Find capitalized words (potential proper nouns / concepts)
        caps = re.findall(r"\b[A-Z][a-z]+\b", text)
        # Find multi-word terms (two+ capitalized words in sequence)
        multi = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b", text)

        topics: list[str] = []
        for term in multi:
            topics.append(term.lower())
        for word in caps:
            w = word.lower()
            if w not in topics:
                topics.append(w)

        # Also extract significant lowercase words (4+ chars)
        words = re.findall(r"\b[a-z]{4,}\b", text.lower())
        for w in words:
            if w not in topics:
                topics.append(w)

        return topics[:10]  # limit to avoid noise

    def _set_knowledge(self, concept: str, level: KnowledgeLevel) -> None:
        """Set the user's knowledge level for a concept.

        Only upgrades (never downgrades) unless explicitly setting
        to NOVICE (from a question).
        """
        concept_lower = concept.lower().strip()
        current = self._model.knowledge.get(concept_lower, KnowledgeLevel.UNKNOWN)

        # Questions set to NOVICE (user is asking → they don't know)
        if level == KnowledgeLevel.NOVICE:
            if current != KnowledgeLevel.EXPERT:
                self._model.knowledge[concept_lower] = level
        else:
            # Upgrade only
            order = {
                KnowledgeLevel.UNKNOWN: 0,
                KnowledgeLevel.NOVICE: 1,
                KnowledgeLevel.FAMILIAR: 2,
                KnowledgeLevel.EXPERT: 3,
            }
            if order.get(level, 0) > order.get(current, 0):
                self._model.knowledge[concept_lower] = level

    def reset(self) -> None:
        """Reset the user model (e.g., for a new conversation partner)."""
        self._model = UserModel()
        self._concept_history.clear()

    # ── Global workspace integration ──────────────────────────────

    def receive_broadcast(self, item: Any) -> None:
        """Receive a broadcast from the global workspace.

        This is **cognitive context for user modeling** — when
        cognitive content ignites the workspace, it becomes context
        that informs how Genesis models the user. In social cognition,
        theory of mind is continuously updated by what's cognitively
        attended to (Frith & Frith, 2006): we infer others' mental
        states from what *we* are currently aware of in the
        interaction.

        The broadcast's topics are stored as cognitive context. When
        ``update_from_user_input`` is next called, these topics
        reinforce the user's knowledge tracking — if the user is
        talking about a topic that's currently cognitive, that
        concept's mention count is incremented, strengthening the
        model's confidence in the user's familiarity with it.

        Args:
            item: A :class:`WorkspaceItem` carrying the broadcast content.
        """
        topics = item.metadata.get("topics") if item.metadata else None
        if not topics:
            return
        for topic in topics:
            if topic and isinstance(topic, str):
                key = topic.lower().strip()
                if key:
                    self._concept_history[key] = self._concept_history.get(key, 0) + 1
