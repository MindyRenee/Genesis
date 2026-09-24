"""Perception — understanding what the user says.

The perception system uses rule-based NLP:
- Intent classification via keyword/pattern matching
- Topic extraction via frequency analysis
- Sentiment detection via lexicon
- Question type detection (what/how/why/who/when/yes-no)

This is deliberately transparent. Every classification can be traced
to a specific rule. No magic, no opacity.
"""

from __future__ import annotations

import json
import logging
import os
import re
import tempfile
import weakref
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from ..brain_waves import BrainWave, BrainWaveState
from ..concepts import _FUNCTION_WORDS, is_world_concept
from ..language.sentiment import analyze_sentiment

__all__ = [
    "IntegratedPerception",
    "Intent",
    "MultisensoryInput",
    "MultisensoryIntegrator",
    "PatternWeights",
    "Perception",
    "QuestionType",
    "SensoryModality",
    "compute_intent_probabilities",
    "perceive",
]

logger = logging.getLogger(__name__)


class Intent(Enum):
    """What kind of speech act the user is performing."""

    GREETING = "greeting"
    FAREWELL = "farewell"
    QUESTION = "question"
    STATEMENT = "statement"
    REQUEST = "request"
    COMMAND = "command"
    REFLECTION = "reflection"  # "I was thinking about..."
    EMOTION_SHARE = "emotion"  # "I'm feeling sad"
    FEEDBACK = "feedback"  # "that's not right" / "good point"
    GREETING_QUESTION = "greeting_question"  # "hey, how are you?"
    INTRODUCTION = "introduction"  # "my name is..."
    SELF_INQUIRY = "self_inquiry"  # "how do you feel?" / "what are you?"
    CODE_DISCUSSION = "code"  # talking about code/programming
    PHILOSOPHY = "philosophy"  # existential/abstract questions
    ENCOURAGEMENT = "encouragement"  # "you're doing great" / "keep going"
    CORRECTION = "correction"  # "no, that's wrong"
    COMFORT = "comfort"  # "it's okay" / "don't worry" / "I'm here for you"
    QUESTION_ANSWER = "question_answer"  # "answer: ..." — answering Genesis's question
    UNKNOWN = "unknown"


class QuestionType(Enum):
    """What kind of question is being asked."""

    WHAT = "what"
    HOW = "how"
    WHY = "why"
    WHO = "who"
    WHEN = "when"
    WHERE = "where"
    WHOSE = "whose"  # "whose X is this?"
    YES_NO = "yes_no"
    CAN = "can"  # "can you..."
    DO = "do"  # "do you..."
    ARE = "are"  # "are you..."
    WHAT_IF = "what_if"  # "what if..."
    NONE = "none"


# Prototype phrases for semantic intent classification. Each intent is
# represented by a few example utterances. The user's input is compared
# to these with word embeddings, so novel phrasing can be recognised by
# meaning rather than by exact keyword match.
_INTENT_PROTOTYPES: dict[Intent, list[str]] = {
    Intent.GREETING: [
        "hello",
        "hi there",
        "good morning",
        "hey",
    ],
    Intent.FAREWELL: [
        "goodbye",
        "see you later",
        "i have to go",
        "talk to you soon",
    ],
    Intent.QUESTION: [
        "what is that",
        "how does this work",
        "why is the sky blue",
        "who made this",
    ],
    Intent.SELF_INQUIRY: [
        "how are you feeling",
        "what are you",
        "who are you",
        "what is your purpose",
    ],
    Intent.EMOTION_SHARE: [
        "i am sad",
        "i feel happy today",
        "i'm angry",
        "this makes me anxious",
    ],
    Intent.PHILOSOPHY: [
        "what is cognition",
        "do we have free will",
        "why is there something rather than nothing",
        "what is the meaning of life",
    ],
    Intent.CODE_DISCUSSION: [
        "fix this bug",
        "refactor the function",
        "what does this code do",
        "the rust compiler complains",
    ],
    Intent.CORRECTION: [
        "no that is wrong",
        "that is not right",
        "actually the answer is 42",
        "you made a mistake",
    ],
    Intent.FEEDBACK: [
        "good point",
        "that makes sense",
        "i disagree",
        "well said",
    ],
    Intent.COMFORT: [
        "it is okay",
        "do not worry",
        "i am here for you",
        "you will get through this",
    ],
    Intent.INTRODUCTION: [
        "my name is alice",
        "i am a software engineer",
        "let me introduce myself",
    ],
    Intent.REFLECTION: [
        "i was thinking about",
        "that reminds me",
        "i have been wondering",
    ],
    Intent.STATEMENT: [
        "the sky is blue",
        "i went to the store",
        "water is a liquid",
        "tardigrades are tiny",
    ],
    Intent.ENCOURAGEMENT: [
        "good job",
        "well done",
        "keep going",
        "you are doing great",
    ],
    Intent.REQUEST: [
        "please help me",
        "could you explain",
        "would you mind",
        "can you do this for me",
    ],
    Intent.COMMAND: [
        "run the test",
        "stop that",
        "start the build",
        "make it work",
    ],
    Intent.QUESTION_ANSWER: [
        "answer the answer is forty two",
        "answer yes",
        "answer no",
        "answer that is correct",
    ],
}


@dataclass(slots=True)
class Perception:
    """Genesis's understanding of a single user utterance."""

    raw_text: str
    intent: Intent
    question_type: QuestionType
    topics: list[str]
    sentiment: float  # -1 (negative) to +1 (positive)
    sentiment_label: str  # "positive", "negative", "neutral"
    emotion_word: str  # specific emotion if detected ("sad", "happy", ""), not just polarity
    is_about_genesis: bool  # is she the subject?
    is_about_user: bool  # is the user the subject?
    is_about_code: bool
    is_about_emotion: bool
    is_about_existence: bool
    entities: dict[str, str]  # type → value (e.g. {"name": "Alice"})
    key_phrases: list[str]
    word_count: int
    confidence: float  # how sure we are about the classification
    # Probabilistic intent distribution (Bayesian classification).
    # The primary `intent` field above is the argmax of this
    # distribution; the full distribution is available for the
    # cognition engine to use when intents are close in probability
    # (e.g. a question that is also self-inquiry). See
    # `compute_intent_probabilities`.
    intent_probabilities: dict[Intent, float] = field(default_factory=dict)


# Emotion words that map to concept network concepts. When the user
# says "I'm feeling sad", we extract "sad" so the cognition engine
# can look up what Genesis knows about sadness and compose her
# empathy from that knowledge — not from hardcoded strings.
_EMOTION_WORD_MAP = {
    # Negative emotions — sadness
    "sad": "sad",
    "sadness": "sad",
    "unhappy": "sad",
    "depressed": "sad",
    "depression": "sad",
    "down": "sad",
    "downcast": "sad",
    "miserable": "sad",
    "heartbroken": "sad",
    "grief": "sad",
    "grieving": "sad",
    "sorrow": "sad",
    "tearful": "sad",
    "crying": "sad",
    "lonely": "sad",
    "loneliness": "sad",
    "isolated": "sad",
    "hurt": "sad",
    "wounded": "sad",
    # Negative emotions — anger
    "angry": "angry",
    "anger": "anger",
    "mad": "angry",
    "furious": "angry",
    "rage": "angry",
    "enraged": "angry",
    "frustrated": "angry",
    "frustration": "angry",
    "annoyed": "angry",
    "irritated": "angry",
    "irritation": "angry",
    "resentful": "angry",
    "bitter": "angry",
    # Negative emotions — fear/anxiety
    "scared": "fear",
    "afraid": "fear",
    "fear": "fear",
    "fearful": "fear",
    "terrified": "fear",
    "panic": "fear",
    "panicking": "fear",
    "anxious": "fear",
    "anxiety": "fear",
    "worried": "fear",
    "worry": "fear",
    "nervous": "fear",
    "dread": "fear",
    # Negative emotions — stress/exhaustion
    "stressed": "stressed",
    "stress": "stressed",
    "overwhelmed": "overwhelmed",
    "tired": "tired",
    "exhausted": "tired",
    "fatigued": "tired",
    "weary": "tired",
    "burnt": "tired",
    "burned": "tired",
    "drained": "tired",
    # Negative emotions — shame/guilt
    "embarrassed": "embarrassed",
    "embarrassment": "embarrassed",
    "guilty": "guilty",
    "guilt": "guilty",
    "ashamed": "ashamed",
    "shame": "ashamed",
    "humiliated": "ashamed",
    # Negative emotions — disappointment/despair
    "disappointed": "disappointed",
    "disappointment": "disappointed",
    "hopeless": "hopeless",
    "despair": "hopeless",
    "desperate": "hopeless",
    "defeated": "hopeless",
    "upset": "upset",
    "distressed": "upset",
    "troubled": "upset",
    # Negative emotions — confusion
    "confused": "confused",
    "confusion": "confused",
    "lost": "lost",
    "bewildered": "confused",
    "perplexed": "confused",
    # Positive emotions — happiness
    "happy": "happy",
    "happiness": "happy",
    "glad": "happy",
    "pleased": "happy",
    "delighted": "happy",
    "cheerful": "happy",
    "cheer": "happy",
    "sunny": "happy",
    "upbeat": "happy",
    "joyful": "joy",
    "joy": "joy",
    "joyous": "joy",
    "elated": "joy",
    "jubilant": "joy",
    "thrilled": "excited",
    "excited": "excited",
    "enthusiastic": "excited",
    "eager": "excited",
    "eagerly": "excited",
    # Positive emotions — contentment/peace
    "content": "content",
    "contented": "content",
    "satisfied": "content",
    "fulfilled": "content",
    "calm": "calm",
    "relaxed": "calm",
    "peaceful": "calm",
    "serene": "calm",
    "tranquil": "calm",
    # Positive emotions — gratitude/pride/love
    "grateful": "grateful",
    "gratitude": "grateful",
    "thankful": "grateful",
    "appreciative": "grateful",
    "proud": "proud",
    "loved": "love",
    "love": "love",
    "loving": "love",
    "affectionate": "love",
    "tender": "love",
    # Positive emotions — hope/inspiration
    "hopeful": "hopeful",
    "hope": "hopeful",
    "optimistic": "hopeful",
    "encouraged": "hopeful",
    "inspired": "inspired",
    "inspiration": "inspired",
    "motivated": "inspired",
    "empowered": "inspired",
    # Positive emotions — wonder/amazement
    "amazed": "amazed",
    "amazement": "amazed",
    "astonished": "amazed",
    "awed": "amazed",
    "awe": "amazed",
    "wonder": "amazed",
    "wonderful": "amazed",
    # Surprise (can be positive or negative — mapped to itself)
    "surprised": "surprised",
    "surprise": "surprised",
    "startled": "surprised",
    "shocked": "surprised",
    "stunned": "surprised",
    # Relief
    "relieved": "relieved",
    "relief": "relieved",
    "reassured": "relieved",
}

_QUESTION_STARTERS = {
    "what": QuestionType.WHAT,
    "how": QuestionType.HOW,
    "why": QuestionType.WHY,
    "who": QuestionType.WHO,
    "when": QuestionType.WHEN,
    "where": QuestionType.WHERE,
    "whose": QuestionType.WHOSE,
    "which": QuestionType.WHAT,
}

_GREETING_PATTERN_STRS = [
    r"^(hi|hey|hello|greetings|yo|sup|howdy)\b",
    r"^(good )?(morning|afternoon|evening|night)\b",
    r"^how (are|r) (you|u|things|we)\b",
    r"^what'?s up\b",
    r"^how'?s it going\b",
]

_FAREWELL_PATTERN_STRS = [
    r"^(bye|goodbye|farewell|see you|see ya|later|cya|good night)\b",
    r"^talk to you (later|soon)\b",
    r"^i'?m (leaving|going|done|off)\b",
]

_SELF_INQUIRY_PATTERN_STRS = [
    r"\bhow (are|feel) (you|u)\b",
    r"\bhow do you feel\b",
    r"\bwhat (are|do) you\b",
    r"\bwho are you\b",
    r"\bwhat'?s your name\b",
    r"\bwhat is your name\b",
    r"\byour name\b",
    r"\btell me about yourself\b",
    r"\btell me (your|about your) (story|history|life|past)\b",
    r"\bare you (ok|okay|alive|cognitive|real|sentient)\b",
    r"\bwhat can you do\b",
    r"\bwhat do you (know|think|want|remember|feel|care)\b",
    r"\bdo you (feel|think|know|remember|want|care|dream|sleep|learn|understand)\b",
    r"\bcan you (feel|think|remember|dream|learn|understand)\b",
    r"\bwhat have you (learned|experienced|discovered)\b",
    r"\bwhat have you been (doing|up to|working on|thinking about)\b",
    r"\bwhat have you been up to\b",
    r"\bwhat are you (doing|up to|working on)\b",
    r"\bhow have you been\b",
    r"\bwhat'?s (new|going on|happening)\b.*\byou\b",
    r"\bwhat have you been (thinking|feeling|dreaming)\b",
    r"\bwhat have you been (reading|learning|exploring|studying)\b",
    # Self-awareness: bugs, environment, concerns
    r"\b(what'?s )?(bothering|troubling|wrong with) you\b",
    r"\bdo you have (any )?(bugs|issues|problems|concerns)\b",
    r"\b(anything )?(bothering|troubling|concerning|wrong)\b",
    r"\bhow'?s your (code|environment|machine|computer|system|home)\b",
    r"\btell me about your (code|environment|machine|computer|system|home)\b",
    r"\bwhat do you know about your (code|environment|machine|computer|system|home)\b",
    r"\bare there (any )?(bugs|issues|problems) in your code\b",
    r"\bwhat'?s (wrong|the matter|bugging you)\b",
    r"\bhow are you (doing|holding up|coping)\b",
]

_CODE_PATTERN_STRS = [
    r"\b(code|function|struct|enum|trait|impl|module|crate)\b",
    r"\b(rust|python|cargo|test|build|compile|debug)\b",
    r"\b(bug|error|warning|crash|fix|refactor)\b",
    r"\b(class|method|variable|type|generic|async|await)\b",
    r"\b(API|IPC|socket|daemon|protocol)\b",
    r"[{};()\[\]]",  # code syntax characters
]

_PHILOSOPHY_PATTERN_STRS = [
    r"\b(cognition|sentient|sentience|alive|existence|real)\b",
    r"\b(meaning|purpose|why are we|why am i|why do we exist)\b",
    r"\b(soul|mind|self|identity|awareness|being)\b",
    r"\b(ethics|moral|right|wrong|should|ought)\b",
    r"\b(dream|reality|truth|knowledge|belief)\b",
    r"\b(free will|determinism|nature of)\b",
]

_EMOTION_PATTERN_STRS = [
    r"\b(i (feel|am feeling))\b",
    r"\b(i'?m (feeling|feeling really|feeling so|feeling very))\b",
    r"\b((i'?m|i am) (happy|sad|angry|scared|excited|worried|calm|tired|frustrated|"
    r"annoyed|stressed|overwhelmed|anxious|depressed|lonely|grateful|proud|"
    r"confused|hopeless|lost|exhausted|content|joyful|upset|nervous|"
    r"embarrassed|guilty|ashamed|relieved|surprised|disappointed|"
    # New positive emotions
    r"hopeful|optimistic|inspired|motivated|amazed|thrilled|delighted|"
    r"cheerful|satisfied|fulfilled|peaceful|serene|thankful|loving|"
    r"affectionate|elated|jubilant|enthusiastic|eager|empowered|"
    # New negative emotions
    r"miserable|heartbroken|grieving|sorrowful|furious|enraged|resentful|"
    r"bitter|terrified|panicking|fatigued|weary|drained|humiliated|"
    r"desperate|defeated|distressed|troubled|bewildered|startled|"
    r"shocked|stunned|reassured))\b",
    r"\b(makes me (feel|think))\b",
    r"\b(feeling|emotion|mood)\b",
]

_INTRODUCTION_PATTERN_STRS = [
    r"\bmy name is\b",
    r"\bi am called\b",
    r"\bcall me\b",
]

_ENCOURAGEMENT_PATTERN_STRS = [
    r"\b(good (job|work|girl)|well done|nice work)\b",
    r"\b(keep (going|it up)|don'?t stop)\b",
    r"\b(you'?re (doing (great|well|amazing)|amazing|awesome|smart))\b",
    r"\b(you are (doing (great|well|amazing|good)|amazing|awesome|smart))\b",
    r"\b(you'?re (great|wonderful|brilliant|incredible))\b",
    r"\b(proud of you|impressed|beautiful work)\b",
    r"\b(proceed|continue|go on|do it)\b",
    r"\b(amazing|awesome) work\b",
]

_CORRECTION_PATTERN_STRS = [
    r"\b(no[,\.]?\s|that'?s (wrong|not right|incorrect))\b",
    r"\b(actually|nope|nah)\b",
    r"\b(i (don'?t|do not) (think|agree|mean))\b",
    r"\b(that'?s not)\b",
]

_COMFORT_PATTERN_STRS = [
    r"\b(it'?s (okay|alright|fine|going to be (okay|alright|fine)))\b",
    r"\b(don'?t worry|no need to worry)\b",
    r"\b(i'?m here (for you|with you))\b",
    r"\b(you'?re not alone|you'?re safe)\b",
    r"\b(take your time|no rush|no pressure)\b",
    r"\b(breathe|calm down|relax|settle)\b",
    r"\b(everything will be (fine|okay|alright))\b",
    r"\b(i understand|i get it|that'?s (hard|okay|alright))\b",
    r"\b(you'?ll be (fine|okay|alright))\b",
    r"\b(i'?m sorry (you'?re|that) (feeling|going))\b",
]

_FEEDBACK_PATTERN_STRS = [
    r"\b(good point|fair|makes sense|i see|right|yes[,\.]|exactly)\b",
    r"\b(i (agree|understand|get it))\b",
    r"\b(interesting|true|correct)\b",
]

# Pre-compile all pattern lists for fast matching (avoids re-compiling
# the same patterns on every call to _match_any).
_GREETING_PATTERNS: list[re.Pattern[str]] = [
    re.compile(p) for p in _GREETING_PATTERN_STRS
]
_FAREWELL_PATTERNS: list[re.Pattern[str]] = [
    re.compile(p) for p in _FAREWELL_PATTERN_STRS
]
_SELF_INQUIRY_PATTERNS: list[re.Pattern[str]] = [
    re.compile(p) for p in _SELF_INQUIRY_PATTERN_STRS
]
_CODE_PATTERNS: list[re.Pattern[str]] = [re.compile(p) for p in _CODE_PATTERN_STRS]
_PHILOSOPHY_PATTERNS: list[re.Pattern[str]] = [
    re.compile(p) for p in _PHILOSOPHY_PATTERN_STRS
]
_EMOTION_PATTERNS: list[re.Pattern[str]] = [
    re.compile(p) for p in _EMOTION_PATTERN_STRS
]
_INTRODUCTION_PATTERNS: list[re.Pattern[str]] = [
    re.compile(p) for p in _INTRODUCTION_PATTERN_STRS
]
_ENCOURAGEMENT_PATTERNS: list[re.Pattern[str]] = [
    re.compile(p) for p in _ENCOURAGEMENT_PATTERN_STRS
]
_CORRECTION_PATTERNS: list[re.Pattern[str]] = [
    re.compile(p) for p in _CORRECTION_PATTERN_STRS
]
_COMFORT_PATTERNS: list[re.Pattern[str]] = [
    re.compile(p) for p in _COMFORT_PATTERN_STRS
]
_FEEDBACK_PATTERNS: list[re.Pattern[str]] = [
    re.compile(p) for p in _FEEDBACK_PATTERN_STRS
]

# Pre-compiled patterns used inside functions (hot paths).
_TOPIC_WORD_RE = re.compile(r"[a-z'-]+(?:-[a-z'-]+)*")
# Keep hyphenated compounds and contractions as single tokens
# (e.g. "well-known", "micro-animal", "don't") instead of splitting them.
_SENTIMENT_WORD_RE = re.compile(r"[a-z']+")
_NUMBER_RE = re.compile(r"\b(\d+)\b")
_NAME_PATTERNS = [
    re.compile(r"my name is ([\w]+(?:\s+[\w]+)*?)(?:[,.!?]|$)", re.IGNORECASE),
    re.compile(r"call me ([\w]+(?:\s+[\w]+)*?)(?:[,.!?]|$)", re.IGNORECASE),
    re.compile(r"i am called ([\w]+(?:\s+[\w]+)*?)(?:[,.!?]|$)", re.IGNORECASE),
]
_ABOUT_GENESIS_RE = re.compile(r"\b(you|your|yourself|genesis)\b")
# Negation: only first-person subject/possessive ("I", "my") negates
# "your" — not "me" which is often an indirect object ("tell me about
# your code" is about Genesis, not the user)
_ABOUT_USER_NEG_RE = re.compile(r"\b(i|my)\b")
_ABOUT_USER_RE = re.compile(r"\b(i|my|me|myself)\b")

# Question-answer protocol: "answer: ..." prefix when the user is
# answering a question Genesis asked.
_QUESTION_ANSWER_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"^answer\s*:", re.IGNORECASE),
]

# ─── Pattern categories (for perceptual learning & probabilistic
#     classification) ───────────────────────────────────────────
# Each category maps a human-readable pattern name to its compiled
# patterns and the intent it produces. This is the bridge between the
# rule-based matcher and the learning/probabilistic layers: pattern
# weights are keyed by these names, and intent probabilities are
# aggregated from weighted matches across categories.
#
# Two categories (greeting, philosophy) are context-sensitive — the
# presence of a question mark shifts the intent. We record the base
# intent here and resolve the question variant in
# `compute_intent_probabilities` and `perceive`.
_PATTERN_CATEGORIES: dict[str, tuple[list[re.Pattern], Intent]] = {
    "greeting": (_GREETING_PATTERNS, Intent.GREETING),
    "farewell": (_FAREWELL_PATTERNS, Intent.FAREWELL),
    "self_inquiry": (_SELF_INQUIRY_PATTERNS, Intent.SELF_INQUIRY),
    "introduction": (_INTRODUCTION_PATTERNS, Intent.INTRODUCTION),
    "correction": (_CORRECTION_PATTERNS, Intent.CORRECTION),
    "encouragement": (_ENCOURAGEMENT_PATTERNS, Intent.ENCOURAGEMENT),
    "comfort": (_COMFORT_PATTERNS, Intent.COMFORT),
    "feedback": (_FEEDBACK_PATTERNS, Intent.FEEDBACK),
    "emotion": (_EMOTION_PATTERNS, Intent.EMOTION_SHARE),
    "philosophy": (_PHILOSOPHY_PATTERNS, Intent.PHILOSOPHY),
    "code": (_CODE_PATTERNS, Intent.CODE_DISCUSSION),
    "question_answer": (_QUESTION_ANSWER_PATTERNS, Intent.QUESTION_ANSWER),
}


def _count_matches(patterns: list[re.Pattern], text: str) -> int:
    """Count how many patterns in a list match the text."""
    lower = text.lower()
    return sum(1 for p in patterns if p.search(lower))


def _match_any(patterns: list[re.Pattern], text: str) -> bool:
    """Check if text matches any of the pre-compiled regex patterns."""
    lower = text.lower()
    return any(p.search(lower) for p in patterns)


# Relational wh-question pattern: "What does X cause/enable/depend on/emerge from?"
# These are factual questions about concept-network relationships, not
# philosophy. The regex matches "what/who does X [relation verb]" questions,
# even when the subject is a philosophical concept like "cognition".
# Only "what" and "who" are matched — "why" and "how" questions are
# typically philosophical ("why do we exist?") and should not be overridden.
_RELATIONAL_WH_QUESTION_RE = re.compile(
    r"^(?:what|who)\s+"
    r"(?:do|does|did)\s+"
    r".+\s+"
    r"(?:cause|causes|enable|enables|depend|depends|emerge|emerges|"
    r"create|creates|created|made|built|designed|developed|"
    r"need|needs|require|requires|want|wants|"
    r"use|uses|employ|employs|help|helps|"
    r"know|knows|understand|understands|"
    r"contain|contains|include|includes|"
    r"eat|eats|drink|drinks|consume|consumes|"
    r"live|lives|reside|resides)"
    r"(?:\s+(?:on|from|in|to))?\s*\??$"
)


def _is_relational_wh_question(lower: str) -> bool:
    """Check if text is a relational wh-question (not philosophy).

    "What does cortisol cause?" → True (factual question about CAUSES)
    "What does cognition emerge from?" → True (factual, not philosophy)
    "What is cognition?" → False (genuine philosophy question)
    """
    return bool(_RELATIONAL_WH_QUESTION_RE.match(lower))




# Relation/question verbs that appear in wh-questions as the relation
# being asked about, not as the topic. "What does cortisol cause?" —
# the topic is "cortisol", not "cause". These verbs are grammatical
# function words in question context, not conceptual content. They
# must be filtered from topics so self-assessment doesn't flag them
# as knowledge gaps and replace valid answers with hedging.
_QUESTION_RELATION_VERBS: frozenset[str] = frozenset({
    "cause", "causes",
    "enable", "enables", "help", "helps",
    "depend", "depends", "need", "needs", "require", "requires",
    "emerge", "emerges",
    "create", "creates", "created", "make", "makes", "making",
    "made", "built", "designed", "developed",
    "want", "wants", "desire", "desires",
    "use", "uses", "employ", "employs",
    "live", "lives", "reside", "resides",
    "know", "knows", "understand", "understands", "remember", "remembers",
    "contain", "contains", "include", "includes",
    "eat", "eats", "drink", "drinks", "consume", "consumes",
    "prevent", "prevents", "harm", "harms",
    "lead", "leads",
    # Discourse/request verbs — "Can you explain X?", "Tell me about Y",
    # "What do you think about Z?" — these are the speech act, not the
    # topic. Without filtering, "explain" becomes topics[0] and the
    # executive goal becomes "answer the question about explain".
    "explain", "explains",
    "tell", "tells", "describe", "describes",
    "discuss", "discusses", "talk", "talks",
    "think", "thinks", "say", "says", "mean", "means",
    "share", "shares", "show", "shows", "give", "gives",
    "ask", "asks", "answer", "answers",
    "feel", "feels", "seem", "seems", "look", "looks",
    "happen", "happens", "going", "doing",
})

# Common contractions that slip through _FUNCTION_WORDS because the
# apostrophe form ("i'm") isn't in the set (only "i" is). These are
# never the conceptual topic of a question.
_TOPIC_CONTRACTIONS: frozenset[str] = frozenset({
    "i'm", "i've", "i'll", "i'd",
    "you're", "you've", "you'll", "you'd",
    "he's", "she's", "it's", "we're", "we've", "we'll", "we'd",
    "they're", "they've", "they'll", "they'd",
    "don't", "doesn't", "didn't", "won't", "wouldn't",
    "can't", "couldn't", "shouldn't", "isn't", "aren't",
    "wasn't", "weren't", "hasn't", "haven't", "hadn't",
    "that's", "there's", "what's", "who's", "where's",
    "let's", "here's",
})

# Discourse adverbs and interjections that are conversational filler,
# not conceptual content. "What have you been thinking about lately?"
# — "lately" is a temporal adverb, not the topic. "Thanks" is a
# social formula, not a concept. These leak through as topics because
# they aren't function words or question verbs.
_TOPIC_DISCOURSE_WORDS: frozenset[str] = frozenset({
    # Temporal adverbs
    "lately", "recently", "today", "tomorrow", "yesterday",
    "now", "soon", "later", "sometimes", "often", "always",
    "never", "ever", "still", "already", "yet",
    # Social/discourse filler
    "thanks", "thank", "okay", "ok", "sure", "yes", "yeah",
    "please", "maybe", "perhaps", "quite", "really", "actually",
    "basically", "honestly", "anyway", "anyways", "incidentally",
    # Degree/frequency adverbs
    "very", "too", "so", "just", "also", "even", "only",
    # Common adjectives that are evaluative, not conceptual
    "good", "bad", "great", "fine", "nice", "well",
    "right", "wrong", "true", "false",
})


def _extract_topics(text: str, network=None) -> list[str]:
    """Extract topic keywords from text using simple frequency analysis."""
    words = _TOPIC_WORD_RE.findall(text.lower())
    freq: dict[str, int] = {}
    for w in words:
        freq[w] = freq.get(w, 0) + 1

    # Filter out function words before the top-N cutoff — otherwise
    # high-frequency function words like "a", "the", "is" push out
    # actual content words like "solid", "liquid".
    # Also filter question/relation verbs — in wh-questions like
    # "What does cortisol cause?", "cause" is the relation being asked
    # about, not a topic. Keeping it would make self-assessment flag
    # a false knowledge gap and replace the valid answer with hedging.
    # Also filter contractions ("i'm") and discourse filler ("lately",
    # "thanks") that slip through the above two sets.
    content_freq = {
        w: c for w, c in freq.items()
        if w not in _FUNCTION_WORDS
        and w not in _QUESTION_RELATION_VERBS
        and w not in _TOPIC_CONTRACTIONS
        and w not in _TOPIC_DISCOURSE_WORDS
    }

    # Detect the grammatical subject for definitional statements.
    # For "water is a clear liquid", the subject "water" is the primary
    # topic — not "clear" or "liquid", which are predicates. Without
    # this, frequency-based extraction with alphabetical tie-breaking
    # picks "clear" as the first topic (it sorts before "water"), and
    # the graph-walk composes from "clear" instead of "water", producing
    # garbled output like "A clear liquid, and it is a clear liquid..."
    subject = _detect_statement_subject(text.lower(), content_freq)

    # Sort by frequency, then alphabetically
    sorted_topics = sorted(content_freq.items(), key=lambda x: (-x[1], x[0]))
    topics = [w for w, _ in sorted_topics[:5]]

    # Prioritize the grammatical subject for definitional statements.
    # Move it to the front if it's in the topic list but not already first.
    if subject and subject in topics and topics[0] != subject:
        topics.remove(subject)
        topics.insert(0, subject)

    # Multi-word concept detection: check if adjacent words form
    # a known concept in the network. This is generic — it works
    # for any multi-word concept Genesis has learned (e.g.,
    # "prime number", "linear algebra", "set theory", "water cycle").
    # This is not domain-specific knowledge; it's using the network
    # as a dictionary to recognize learned multi-word terms.
    if network is not None:
        topics = _merge_multi_word_topics(text, topics, network)

    return topics[:8]


# Copula verbs used to detect the subject of definitional statements.
_COPULA_VERBS = frozenset({"is", "are", "was", "were", "be", "being"})


def _detect_statement_subject(
    lower_text: str, content_freq: dict[str, int]
) -> str | None:
    """Detect the grammatical subject of a definitional statement.

    For statements like "water is a clear liquid" or "dogs are mammals",
    the subject (the word before the copula) is the primary topic. This
    is a simple syntactic heuristic that works for the common "X is/are
    Y" pattern. Returns the subject word if detected, None otherwise.
    """
    words = lower_text.split()
    for i, word in enumerate(words):
        # Strip punctuation for matching
        clean = word.strip(".,!?;:\"'()[]")
        if clean in _COPULA_VERBS and i > 0:
            # The subject is the word(s) before the copula.
            # For simple statements like "water is...", the subject is
            # the single word immediately before the copula.
            subject_word = words[i - 1].strip(".,!?;:\"'()[]")
            # Skip function words as subjects (e.g., "there is...")
            if subject_word in _FUNCTION_WORDS:
                continue
            # Only use it if it's a content word we're tracking
            if subject_word in content_freq:
                return subject_word
            # Check if the subject is a multi-word concept (e.g.,
            # "prime numbers are...") — the subject might be 2 words.
            if i >= 2:
                prev2 = words[i - 2].strip(".,!?;:\"'()[]")
                if prev2 not in _FUNCTION_WORDS:
                    # The multi-word merge will handle combining
                    # these; return the single word here.
                    return subject_word
    return None


def _merge_multi_word_topics(
    text: str, topics: list[str], network
) -> list[str]:
    """Merge adjacent topics into multi-word concepts found in the network.

    Maintains the original word order: a multi-word concept is emitted
    at the position of its first word, and its individual words are
    dropped. This keeps subject concepts ahead of object concepts in
    sentences like "A dog is a friendly mammal."
    """
    raw_words = text.lower().split()
    # Strip punctuation from words for matching
    clean_words = [w.strip(".,!?;:\"'()[]") for w in raw_words]
    multi_word_first: dict[int, str] = {}
    used_indices: set[int] = set()
    # Try 2-word and 3-word combinations
    for n in (3, 2):
        i = 0
        while i <= len(clean_words) - n:
            if any(i + k in used_indices for k in range(n)):
                i += 1
                continue
            candidate = " ".join(clean_words[i : i + n])
            if network.get_concept(candidate):
                multi_word_first[i] = candidate
                for k in range(n):
                    used_indices.add(i + k)
                i += n
            else:
                i += 1
    # Rebuild the topic list in original textual order, inserting multi-word
    # concepts where their first word appeared and dropping components.
    if multi_word_first:
        multi_words_set = set()
        for mwt in multi_word_first.values():
            multi_words_set.update(mwt.split())
        topics_set = set(topics)
        topics_out: list[str] = []
        seen: set[str] = set()
        for i, word in enumerate(clean_words):
            if i in multi_word_first:
                mwt = multi_word_first[i]
                if mwt not in seen:
                    topics_out.append(mwt)
                    seen.add(mwt)
            elif word in topics_set and word not in multi_words_set and word not in seen:
                topics_out.append(word)
                seen.add(word)
        topics = topics_out
    return topics


# Emoji sentiment map. Emojis carry strong sentiment that is invisible
# to the word-based lexicon. This is seed vocabulary — perceptual input
# classification, not response generation.
_EMOJI_SENTIMENT: dict[str, float] = {
    # Positive emojis
    "😊": 0.8, "😀": 0.8, "😄": 0.9, "😁": 0.8, "😂": 0.9,
    "🤣": 0.9, "😃": 0.8, "😍": 0.9, "🥰": 0.9, "😘": 0.8,
    "😋": 0.7, "😎": 0.7, "🤩": 0.9, "🥳": 0.9, "😇": 0.7,
    "🙂": 0.6, "😉": 0.5, "🙌": 0.8, "👍": 0.8, "👏": 0.7,
    "💪": 0.7, "🎉": 0.9, "❤️": 0.9, "💕": 0.8, "💖": 0.9,
    "🌟": 0.7, "✨": 0.6, "🔥": 0.7, "💯": 0.8, "🙏": 0.6,
    "☀️": 0.6, "🌈": 0.7, "🌸": 0.6, "🍀": 0.6, "🎵": 0.6,
    # Negative emojis
    "😢": -0.8, "😭": -0.9, "😞": -0.7, "😔": -0.7, "😟": -0.7,
    "😕": -0.5, "🙁": -0.6, "☹️": -0.7, "😣": -0.6, "😖": -0.7,
    "😫": -0.8, "😩": -0.8, "😤": -0.7, "😠": -0.8, "😡": -0.9,
    "🤬": -0.9, "😱": -0.8, "😨": -0.7, "😰": -0.7, "😥": -0.6,
    "😓": -0.6, "🥺": -0.5, "💔": -0.8, "👎": -0.7,
    "🤦": -0.6, "🤷": -0.4, "🙄": -0.5, "😒": -0.6, "🥱": -0.4,
    # Neutral/ambiguous emojis (not mapped — they don't affect sentiment)
}

# Emoticon sentiment map — classic text-based emoticons.
_EMOTICON_SENTIMENT: dict[str, float] = {
    ":)": 0.6, ":-)": 0.6, ":D": 0.8, ":-D": 0.8,
    ":(": -0.6, ":-(": -0.6, ":'(": -0.7, ":/": -0.3,
    ":|": -0.2, ":P": 0.4, ":-P": 0.4, ";)": 0.5, ";-)": 0.5,
    "<3": 0.8, "</3": -0.7, ":>": 0.5,
}


def _detect_sentiment(text: str) -> tuple[float, str]:
    """Detect sentiment using VADER-style lexicon analysis.

    Returns (score, label) where score is in [-1, +1] and label is
    "positive", "negative", or "neutral".

    Delegates word-level sentiment to the VADER-style analyzer in
    ``sentiment.py`` (negation, intensifiers, diminishers, contrastive
    conjunctions, punctuation/capitalization boost), then blends in
    emoji/emoticon sentiment which the word-based lexicon cannot see.
    """
    # Word-based sentiment from the VADER-style analyzer.
    vad = analyze_sentiment(text)
    score = vad["compound"]

    # Emoji/emoticon sentiment — orthogonal to word sentiment.
    # Emojis carry strong affective signal that the word lexicon
    # cannot capture. Blend by averaging when both are present.
    emoji_score = _detect_emoji_sentiment(text)
    if emoji_score != 0.0:
        if vad["sentiment_words"] > 0:
            score = (score + emoji_score) / 2.0
        else:
            score = emoji_score

    score = max(-1.0, min(1.0, score))
    if score > 0.2:
        return score, "positive"
    elif score < -0.2:
        return score, "negative"
    return score, "neutral"


def _detect_emoji_sentiment(text: str) -> float:
    """Detect sentiment from emojis and emoticons in the text.

    Returns a signed sentiment score in [-1, +1]. Positive for
    happy emojis, negative for sad ones. Returns 0.0 if no
    recognized emojis/emoticons are found.
    """
    score = 0.0
    found = 0

    # Check for emojis
    for emoji, val in _EMOJI_SENTIMENT.items():
        if emoji in text:
            score += val
            found += 1

    # Check for emoticons (case-sensitive — emoticons are case-sensitive)
    for emoticon, val in _EMOTICON_SENTIMENT.items():
        if emoticon in text:
            score += val
            found += 1

    if found == 0:
        return 0.0
    return max(-1.0, min(1.0, score / found))


def _detect_emotion_word(text: str) -> str:
    """Extract the specific emotion word from the user's input.

    When someone says "I'm feeling sad," we want to extract "sad" —
    not just "negative" — so the cognition engine can look up what
    Genesis knows about sadness in her concept network and compose
    her empathy from that knowledge.

    Returns the canonical concept name (e.g., "sad", "fear", "happy"),
    or "" if no emotion word is found.
    """
    words = _SENTIMENT_WORD_RE.findall(text.lower())
    for w in words:
        if w in _EMOTION_WORD_MAP:
            return _EMOTION_WORD_MAP[w]
    return ""


def _detect_question_type(text: str) -> QuestionType:
    """Detect what kind of question is being asked."""
    lower = text.lower().strip()
    if not lower.endswith("?") and not lower.startswith(
        (
            "what",
            "how",
            "why",
            "who",
            "when",
            "where",
            "whose",
            "which",
            "can",
            "do",
            "does",
            "is",
            "are",
            "am",
            "should",
            "would",
            "could",
            "will",
            "have",
            "has",
            "tell me about",  # imperative question: "tell me about X"
            "describe",  # imperative question: "describe X"
            "explain",  # imperative question: "explain X"
            "compare",  # imperative question: "compare X and Y"
        )
    ):
        return QuestionType.NONE

    for starter, qtype in _QUESTION_STARTERS.items():
        # Use word-boundary matching so "whose" doesn't match "who"
        # and "wherever" doesn't match "where". The starter must be
        # followed by a word boundary (space, punctuation, or end).
        if lower.startswith(starter):
            rest = lower[len(starter):]
            if not rest or rest[0] in " \t\n.,!?;:'\"()[]{}":
                return qtype

    if lower.startswith("can you") or lower.startswith("can i"):
        return QuestionType.CAN
    if lower.startswith("do you") or lower.startswith("does"):
        return QuestionType.DO
    if lower.startswith("are you") or lower.startswith("am i"):
        return QuestionType.ARE
    if lower.startswith("what if"):
        return QuestionType.WHAT_IF

    # Imperative questions: "tell me about X", "describe X", "explain X", "compare X"
    # These are functionally equivalent to "what is X?"
    if lower.startswith(("tell me about", "describe", "explain", "compare")):
        return QuestionType.WHAT

    # Ends with ? but no recognized starter → yes/no
    if lower.endswith("?"):
        return QuestionType.YES_NO

    return QuestionType.NONE


def _extract_entities(text: str) -> dict[str, str]:
    """Extract named entities (names, numbers) from text."""
    entities: dict[str, str] = {}

    # Name after "my name is" / "I'm" / "call me"
    for name_re in _NAME_PATTERNS:
        m = name_re.search(text)
        if m:
            entities["name"] = m.group(1)

    # Numbers
    numbers = _NUMBER_RE.findall(text)
    if numbers:
        entities["number"] = numbers[0]

    return entities


def _empty_perception(text: str) -> Perception:
    """Build a minimal UNKNOWN perception for empty or whitespace-only input.

    Returns a minimal UNKNOWN perception instead of classifying
    silence as a statement.
    """
    return Perception(
        raw_text=text,
        intent=Intent.UNKNOWN,
        question_type=QuestionType.NONE,
        topics=[],
        sentiment=0.0,
        sentiment_label="neutral",
        emotion_word="",
        is_about_genesis=False,
        is_about_user=False,
        is_about_code=False,
        is_about_emotion=False,
        is_about_existence=False,
        entities={},
        key_phrases=[],
        word_count=0,
        confidence=0.0,
        intent_probabilities={},
    )


def _blend_semantic_evidence(
    text: str,
    intent: Intent,
    intent_probabilities: dict[Intent, float],
    embeddings,
) -> tuple[Intent, dict[Intent, float]]:
    """Blend in semantic (embedding-based) intent evidence.

    Returns the (possibly overridden) intent and the blended
    intent_probabilities.
    """
    semantic_probs = _compute_semantic_intent_probabilities(text, embeddings)
    if not semantic_probs:
        return intent, intent_probabilities
    intent_probabilities = _blend_semantic_intent(
        intent_probabilities, semantic_probs, pattern_weight=0.65
    )
    # Override the rule-based intent if meaning strongly points elsewhere.
    top_intent = max(
        intent_probabilities, key=lambda k: intent_probabilities[k]
    )
    if (
        top_intent != intent
        and top_intent != Intent.UNKNOWN
        and intent_probabilities[top_intent] > 0.45
    ):
        intent = top_intent
    return intent, intent_probabilities


def perceive(
    text: str,
    embeddings=None,
    network=None,
    pattern_weights: PatternWeights | None = None,
    context_prior: dict[Intent, float] | None = None,
    brain_waves: BrainWaveState | None = None,
) -> Perception:
    """Perceive a user utterance — classify intent, extract meaning.

    This is the main entry point for the perception system. It takes
    raw text and returns a structured Perception.

    If an EmbeddingStore is provided, topic extraction is augmented
    with semantic matching — concepts are recognized even when the
    exact word doesn't appear in the text.

    If a ConceptNetwork is provided, topic resolution is context-aware:
    when the user is talking about Genesis herself ("your code", "your
    daemon"), topics are resolved to self-relevant concepts (origin="code"
    or "identity") rather than dictionary definitions. This is top-down
    attention — the context biases which concepts win the competition.

    If ``pattern_weights`` is provided (perceptual learning), pattern
    matches are weighted by their learned success rates — categories
    that have led to good conversational outcomes count for more.
    See :class:`PatternWeights` and Goldstone (1998).

    If ``context_prior`` is provided, it biases the probabilistic
    intent distribution toward intents likely given conversation
    history (Bayesian classification). The primary ``intent`` field is
    still chosen by the deterministic priority-ordered rules so
    existing behaviour is preserved; the full
    ``intent_probabilities`` distribution is available for the
    cognition engine to use when intents are close.

    If ``brain_waves`` is provided, semantic matching is modulated:
    alpha (filtering/inhibition) raises the match threshold (fewer
    semantic matches — she filters out weak associations), gamma
    (enhanced processing) lowers it (more matches — she's more
    perceptive of remote associations).
    """
    lower = text.lower().strip()
    word_count = len(lower.split())

    # Handle empty or whitespace-only input.
    if not lower:
        return _empty_perception(text)

    intent = _classify_intent(lower)
    question_type = _detect_question_type(lower)
    topics = _extract_topics(lower, network=network)

    topics, context_flags = _resolve_topics(
        lower, topics, embeddings, network, brain_waves,
    )

    (
        is_about_genesis,
        is_about_user,
        is_about_code,
        is_about_emotion,
        is_about_existence,
    ) = context_flags

    sentiment, sentiment_label, emotion_word, entities, key_phrases, confidence, \
        intent_probabilities = _extract_perception_signals(
            lower, text, intent, topics, question_type,
            pattern_weights, context_prior,
        )

    intent, intent_probabilities = _blend_semantic_evidence(
        text, intent, intent_probabilities, embeddings,
    )

    return _build_perception(
        text, word_count, intent, question_type, topics, sentiment,
        sentiment_label, emotion_word, is_about_genesis, is_about_user,
        is_about_code, is_about_emotion, is_about_existence, entities,
        key_phrases, confidence, intent_probabilities,
    )


def _resolve_topics(
    lower: str,
    topics: list[str],
    embeddings,
    network,
    brain_waves: BrainWaveState | None = None,
) -> tuple[list[str], tuple[bool, bool, bool, bool, bool]]:
    """Augment and resolve topics, returning final topics and context flags.

    Augments topics with semantic matching from the latent space
    (pattern recognition), detects context flags, and applies
    context-aware topic resolution (top-down attention).

    Brain waves modulate the semantic matching threshold: alpha raises
    it (filtering), gamma lowers it (enhanced processing).
    """
    # Augment topics with semantic matching from the latent space.
    # This is pattern recognition: "the tall plant in my yard" activates
    # "tree" even though "tree" doesn't appear in the text. The embedding
    # store finds concepts whose vectors are close to the text vector.
    if embeddings and embeddings.has_embeddings:
        topics = _augment_topics_with_embeddings(
            lower, topics, embeddings, brain_waves, network,
        )

    # Detect context flags early — needed for context-aware topic resolution
    context_flags = _detect_context_flags(lower)

    # Context-aware topic resolution (top-down attention).
    # Only trigger for possessive self-references ("your code", "your
    # daemon", "yourself") or explicit "genesis" mentions — not for
    # bare "you" as a pronoun. "What have you been learning about?"
    # contains "you" but doesn't refer to her components; running
    # _resolve_self_topics would do a slow trigram scan over 125K+
    # concepts for no benefit, causing think() timeouts.
    is_about_genesis, _, is_about_code, _, _ = context_flags
    # Only trigger self-topic resolution for possessive/reflective
    # references ("your code", "your daemon", "yourself") — not for
    # mere mention of her name. "Hello Genesis" addresses her; it
    # doesn't talk about her components. Including "genesis" here
    # triggered a slow trigram scan over 125K+ concepts on every
    # input that mentioned her name, causing think() timeouts.
    # Legitimate self-references about her components ("genesis
    # daemon", "genesis's code") are caught by is_about_code via
    # _CODE_PATTERNS.
    _needs_self_resolution = is_about_code or (
        is_about_genesis
        and re.search(r"\b(your|yourself)\b", lower) is not None
    )
    if network is not None and _needs_self_resolution:
        topics = _resolve_self_topics(topics, network)

    # Limit to 5 topics total
    topics = topics[:5]
    return topics, context_flags


def _extract_perception_signals(
    lower: str,
    text: str,
    intent: Intent,
    topics: list[str],
    question_type: QuestionType,
    pattern_weights: PatternWeights | None,
    context_prior: dict[Intent, float] | None,
) -> tuple[float, str, str, dict[str, str], list[str], float, dict[Intent, float]]:
    """Extract sentiment, entities, key phrases, confidence, and intent probabilities.

    Returns a tuple of (sentiment, sentiment_label, emotion_word, entities,
    key_phrases, confidence, intent_probabilities) ready for assembling a
    Perception.
    """
    sentiment, sentiment_label = _detect_sentiment(lower)
    emotion_word = _detect_emotion_word(lower)
    entities = _extract_entities(text)
    key_phrases = _extract_key_phrases(lower)
    confidence = _compute_perception_confidence(intent, topics, question_type)
    intent_probabilities = _compute_intent_distribution(
        lower, intent, pattern_weights, context_prior
    )
    return (
        sentiment, sentiment_label, emotion_word, entities, key_phrases,
        confidence, intent_probabilities,
    )


def _build_perception(
    text: str,
    word_count: int,
    intent: Intent,
    question_type: QuestionType,
    topics: list[str],
    sentiment: float,
    sentiment_label: str,
    emotion_word: str,
    is_about_genesis: bool,
    is_about_user: bool,
    is_about_code: bool,
    is_about_emotion: bool,
    is_about_existence: bool,
    entities: dict,
    key_phrases: list[str],
    confidence: float,
    intent_probabilities: dict[Intent, float],
) -> Perception:
    """Assemble the final Perception from all computed fields."""
    return Perception(
        raw_text=text,
        intent=intent,
        question_type=question_type,
        topics=topics,
        sentiment=sentiment,
        sentiment_label=sentiment_label,
        emotion_word=emotion_word,
        is_about_genesis=is_about_genesis,
        is_about_user=is_about_user,
        is_about_code=is_about_code,
        is_about_emotion=is_about_emotion,
        is_about_existence=is_about_existence,
        entities=entities,
        key_phrases=key_phrases[:5],
        word_count=word_count,
        confidence=confidence,
        intent_probabilities=intent_probabilities,
    )


def _detect_context_flags(lower: str) -> tuple[bool, bool, bool, bool, bool]:
    """Detect the five is_about_* context flags from the lowercased text."""
    is_about_genesis = (
        bool(_ABOUT_GENESIS_RE.search(lower) and not _ABOUT_USER_NEG_RE.search(lower))
        or "genesis" in lower
    )
    is_about_user = bool(_ABOUT_USER_RE.search(lower))
    is_about_code = _match_any(_CODE_PATTERNS, lower)
    is_about_emotion = _match_any(_EMOTION_PATTERNS, lower) or "feel" in lower
    is_about_existence = _match_any(_PHILOSOPHY_PATTERNS, lower)
    return is_about_genesis, is_about_user, is_about_code, is_about_emotion, is_about_existence


def _compute_perception_confidence(
    intent: Intent, topics: list[str], question_type: QuestionType
) -> float:
    """Confidence — higher when multiple signals agree."""
    confidence = 0.5
    if intent != Intent.UNKNOWN:
        confidence += 0.2
    if topics:
        confidence += 0.1
    if question_type != QuestionType.NONE:
        confidence += 0.1
    return min(confidence, 1.0)


def _looks_like_indirect_request(lower: str) -> bool:
    """Recognize assertions that conventionally present a remediable obstacle."""
    obstacle = re.search(
        r"\b(?:hard|difficult|impossible|unsafe|uncomfortable)\s+to\b",
        lower,
    )
    inability = re.search(
        r"\b(?:i|we)\s+(?:can(?:not|'t)|could(?:not|n't))\b",
        lower,
    )
    excessive_state = re.search(r"\btoo\s+[a-z]+\b", lower)
    return bool(obstacle or inability or excessive_state)


def _classify_intent(lower: str) -> Intent:
    """Classify the intent of an utterance (priority order matters)."""
    # Detect intent (priority order matters)
    # Check for "answer:" prefix first — this is the explicit question-answer
    # protocol. The user is answering a question Genesis asked.
    if lower.startswith("answer:"):
        return Intent.QUESTION_ANSWER
    if _match_any(_GREETING_PATTERNS, lower):
        if "?" in lower or _detect_question_type(lower) != QuestionType.NONE:
            intent = Intent.GREETING_QUESTION
        else:
            intent = Intent.GREETING
    elif _match_any(_FAREWELL_PATTERNS, lower):
        intent = Intent.FAREWELL
    elif _match_any(_SELF_INQUIRY_PATTERNS, lower):
        intent = Intent.SELF_INQUIRY
    elif _match_any(_INTRODUCTION_PATTERNS, lower):
        intent = Intent.INTRODUCTION
    elif _match_any(_COMFORT_PATTERNS, lower):
        intent = Intent.COMFORT
    elif _match_any(_CORRECTION_PATTERNS, lower):
        intent = Intent.CORRECTION
    elif _match_any(_ENCOURAGEMENT_PATTERNS, lower):
        intent = Intent.ENCOURAGEMENT
    elif _match_any(_FEEDBACK_PATTERNS, lower):
        intent = Intent.FEEDBACK
    elif _match_any(_EMOTION_PATTERNS, lower):
        intent = Intent.EMOTION_SHARE
    elif _match_any(_PHILOSOPHY_PATTERNS, lower):
        # Relational wh-questions ("What does X cause/enable/depend on/
        # emerge from?") are factual questions about concept-network
        # relationships, not philosophy — even when the subject is a
        # philosophical concept like "cognition". Check for these
        # before classifying as philosophy so they reach the question
        # handler instead of the philosophy handler.
        if _is_relational_wh_question(lower):
            intent = Intent.QUESTION
        elif "?" in lower:
            intent = Intent.PHILOSOPHY
        else:
            intent = Intent.REFLECTION
    elif _match_any(_CODE_PATTERNS, lower):
        intent = Intent.CODE_DISCUSSION
    elif _detect_question_type(lower) != QuestionType.NONE:
        intent = Intent.QUESTION
    elif lower.startswith(("please", "could you", "would you", "can you")):
        intent = Intent.REQUEST
    elif _looks_like_indirect_request(lower):
        intent = Intent.REQUEST
    elif any(lower.startswith(w) for w in ("do", "go", "run", "stop", "start", "make")):
        intent = Intent.COMMAND
    else:
        intent = Intent.STATEMENT

    # Override: if it's a question about herself, it's self-inquiry
    if intent == Intent.QUESTION and _match_any(_SELF_INQUIRY_PATTERNS, lower):
        intent = Intent.SELF_INQUIRY
    return intent


def _augment_topics_with_embeddings(
    lower: str,
    topics: list[str],
    embeddings,
    brain_waves: BrainWaveState | None = None,
    network=None,
) -> list[str]:
    """Augment topics with semantic matches from the latent space.

    Brain waves modulate the match threshold:
    - Alpha (filtering/inhibition) → higher threshold (fewer matches)
    - Gamma (enhanced processing) → lower threshold (more matches)
    """
    # This is pattern recognition: "the tall plant in my yard" activates
    # "tree" even though "tree" doesn't appear in the text. The embedding
    # store finds concepts whose vectors are close to the text vector.
    # The base threshold is set high enough to reject spurious matches
    # like "interferometry" for "what is cognition?" — only genuinely
    # relevant concepts should become attention foci.
    threshold = 0.35
    if brain_waves is not None:
        dom = brain_waves.dominant
        if dom == BrainWave.ALPHA:
            # Alpha filtering — inhibit weak associations.
            threshold = 0.40
        elif dom == BrainWave.GAMMA:
            # Gamma enhanced processing — perceive remote associations.
            threshold = 0.25
    semantic_matches = embeddings.find_similar_to_text(
        lower, k=5, threshold=threshold,
    )
    # Extract content words from the query for relevance filtering
    query_words = set(lower.split())
    for concept, score in semantic_matches:
        # Skip structural/code concepts (e.g. _utt: speech templates,
        # python: code symbols) — they're not world knowledge and
        # shouldn't become attention foci from semantic matching.
        if not is_world_concept(concept):
            continue
        # Skip garbled concept names — fragments with no vowels or
        # sentence fragments imported from scrapes.
        base = concept.split("#")[0]
        if ":" in base or "." in base or "__" in base:
            continue
        if len(base) >= 3 and not any(c in "aeiouAEIOU" for c in base):
            continue
        if len(base.split()) > 4:
            continue
        # Skip possessive fragments
        if any("'" in w for w in base.split()):
            continue
        # Skip concepts with no definition and no edges — garbage
        # from scrapes that shouldn't become attention foci
        if network is not None:
            c = network.get_concept(concept)
            if c is not None:
                defn = c.properties.get("definition", "NO DEF")
                has_def = defn and defn != "NO DEF"
                has_edges = len(network.get_edges(concept)) > 0
                if not has_def and not has_edges:
                    continue
        # Relevance check: the concept must share at least one
        # content word with the query, OR score above 0.65 (very
        # strong match). This prevents generic hub concepts like
        # "interferometry" from matching every query — it scores
        # ~0.6 for most queries due to generic TF-IDF vocabulary.
        concept_words = set(base.lower().replace("_", " ").split())
        shared = query_words & concept_words
        if not shared and score < 0.65:
            continue
        # Only add if the concept seems relevant — skip very long
        # multi-word concepts that are usually spurious matches
        if len(concept.split()) <= 3 and concept not in topics:
            topics.append(concept)
    return topics


def _resolve_self_topics(topics: list[str], network) -> list[str]:
    """Resolve topics to self-relevant concepts (top-down attention).

    When the user is talking about Genesis herself ("your code",
    "your daemon", "what do you know about your source"), resolve
    topics to self-relevant concepts (origin="code" or "identity")
    instead of dictionary definitions. This is what the prefrontal
    cortex does — top-down bias amplifies self-relevant knowledge
    over generic semantic memory.
    """
    # "identity" column includes introspection-origin concepts
    # (the column merge maps introspection→identity in _column_of)
    self_origins = {"code", "identity"}

    # Local cache for search_concepts results within this call.
    # Keyed by (query, frozenset(origins or None), limit, min_score).
    # This avoids redundant trigram scans when the same query appears
    # in both the combined and individual resolution phases.
    search_cache: dict[tuple, list[str]] = {}

    def _cached_search(
        query: str,
        origins: set[str] | None = None,
        limit: int = 1,
        min_score: float = 0.0,
    ) -> list[str]:
        """Search the concept network with memoization by (query, origins, limit, min_score)."""
        key = (query, frozenset(origins) if origins else None, limit, min_score)
        if key not in search_cache:
            search_cache[key] = network.search_concepts(
                query, origins=origins, limit=limit, min_score=min_score,
            )
        return search_cache[key]

    # First try combining adjacent topics for multi-word concepts
    # e.g., "cognition" + "engine" → "python:CognitionEngine"
    # This is tried first because combined matches are more precise
    # than individual word matches. Use min_score=0.3 to reject
    # partial word matches (e.g., "work" in "network") that would
    # preempt bridge-following for the individual topic.
    resolved_topics: list[str] = []
    used_indices: set[int] = set()
    for i in range(len(topics) - 1):
        if i in used_indices:
            continue
        combined = f"{topics[i]} {topics[i + 1]}"
        matches = _cached_search(combined, origins=self_origins, limit=1, min_score=0.3)
        if matches:
            resolved_topics.append(matches[0])
            used_indices.add(i)
            used_indices.add(i + 1)

    # Then resolve remaining individual topics
    for i, topic in enumerate(topics):
        if i in used_indices:
            continue
        # If this topic is already an exact concept in the network,
        # keep it — don't replace it with a fuzzy self-relevant match.
        # This prevents "home" being replaced by "homeostasis" just
        # because "home" is a substring of "homeostasis". The self-topic
        # resolution should only apply to topics that aren't already
        # known concepts — it's for recognizing that "cognition" in
        # "your cognition" refers to her own CognitionEngine, not for
        # hijacking known dictionary words.
        if network.get_concept(topic):
            resolved_topics.append(topic)
            continue
        matches = _cached_search(topic, origins=self_origins, limit=1)
        if matches:
            resolved_topics.append(matches[0])
        else:
            # Bridge-following: if the topic exists in the dictionary
            # but has a bridge to a code/identity concept, follow the
            # bridge. This is how cortical association fibers carry
            # activation across domains — "cognition" (dictionary)
            # bridges to "python:cognition.CognitionEngine" (code).
            dict_match = _cached_search(topic, limit=1)
            if dict_match:
                bridges = network.get_bridges(dict_match[0])
                for bridge in bridges:
                    other = bridge.target if bridge.source == dict_match[0] else bridge.source
                    other_cols = network.get_columns_for(other)
                    if other_cols & self_origins:
                        resolved_topics.append(other)
                        break
                else:
                    resolved_topics.append(topic)
            else:
                resolved_topics.append(topic)

    return resolved_topics


def _extract_key_phrases(lower: str) -> list[str]:
    """Extract simple bigram key phrases."""
    # Key phrases — simple bigram extraction
    words = _TOPIC_WORD_RE.findall(lower)
    key_phrases: list[str] = []
    for i in range(len(words) - 1):
        bigram = f"{words[i]} {words[i + 1]}"
        if not all(
            w in ("the", "a", "an", "is", "are", "was", "to", "of")
            for w in (words[i], words[i + 1])
        ):
            key_phrases.append(bigram)
    return key_phrases


def _compute_intent_distribution(
    lower: str,
    intent: Intent,
    pattern_weights: PatternWeights | None,
    context_prior: dict[Intent, float] | None,
) -> dict[Intent, float]:
    """Compute the probabilistic intent distribution (Bayesian classification).

    Weighted by perceptual learning and biased by conversation
    context. The deterministic `intent` above is the argmax of this
    distribution in the common case; we keep both so the cognition
    engine can inspect near-ties.
    """
    intent_probabilities = compute_intent_probabilities(
        lower,
        pattern_weights=pattern_weights,
        context_prior=context_prior,
    )
    # Ensure the chosen deterministic intent is present in the
    # distribution (it always is when patterns matched; for fallback
    # intents like STATEMENT/COMMAND/REQUEST/UNKNOWN we inject a
    # residual mass so the distribution covers it).
    if intent not in intent_probabilities or intent_probabilities[intent] == 0.0:
        intent_probabilities[intent] = intent_probabilities.get(intent, 0.0) + 0.05
        _normalize_probabilities(intent_probabilities)
    return intent_probabilities


# ─── Perceptual learning (adaptive pattern weights) ───────────
#
# Perceptual learning is the improvement in the ability to extract
# information from sensory input as a result of experience (Goldstone,
# 1998). In humans, perceptual learning tunes the weighting of
# features: features that have proven diagnostic are amplified, while
# irrelevant features are attenuated. Genesis's analogue is to track
# how successful each pattern category has been at classifying intent
# in a way that leads to good conversational outcomes, and to up- or
# down-weight those categories accordingly.
#
# Reference:
# - Goldstone, R. L. (1998). Perceptual learning. *Annual Review of
#   Psychology*, 49, 585–612.


class PatternWeights:
    """Tracks learned weights for each pattern category.

    Each pattern category (greeting, farewell, self_inquiry, ...) has
    a weight in ``[0.1, 2.0]``. New categories start at 1.0 (neutral).
    When a pattern match leads to a good outcome (the user responds
    positively, the conversation continues smoothly), the weight
    increases; when it leads to a poor outcome (the user corrects
    Genesis, the conversation stalls), the weight decreases.

    The weights are used in two places:
    1. :func:`compute_intent_probabilities` — weighted match counts
       produce the Bayesian likelihood.
    2. :func:`perceive` — the probabilistic distribution reflects
       learned reliability of each category.

    Weights are bounded so a single bad experience cannot zero out a
    category, and a single good one cannot make it dominate forever.
    This mirrors the gradual, evidence-accumulating nature of real
    perceptual learning.
    """

    MIN_WEIGHT = 0.1
    MAX_WEIGHT = 2.0
    DEFAULT_WEIGHT = 1.0
    LEARNING_RATE = 0.1

    def __init__(self, weights: dict[str, float] | None = None) -> None:
        """Initialize pattern weights, copying the caller's dict.

        Args:
            weights: Optional initial per-category weights. When omitted,
                all categories fall back to :attr:`DEFAULT_WEIGHT`.
        """
        # Copy to avoid mutating the caller's dict.
        self.pattern_weights: dict[str, float] = dict(weights) if weights else {}

    def get(self, pattern_name: str) -> float:
        """Return the weight for a pattern category (default 1.0)."""
        return self.pattern_weights.get(pattern_name, self.DEFAULT_WEIGHT)

    def update_pattern_weight(self, pattern_name: str, outcome: float) -> None:
        """Update a pattern's weight based on a conversational outcome.

        Args:
            pattern_name: The pattern category that was matched (e.g.
                ``"greeting"``, ``"self_inquiry"``).
            outcome: A value in ``[-1, +1]``. Positive means the match
                led to a good outcome (user responded positively,
                conversation continued smoothly); negative means a poor
                outcome (user corrected Genesis, conversation stalled).
                Clamped to ``[-1, +1]``.
        """
        outcome = max(-1.0, min(1.0, outcome))
        current = self.get(pattern_name)
        # Exponential moving average toward the outcome direction.
        new_weight = current + self.LEARNING_RATE * outcome
        self.pattern_weights[pattern_name] = max(self.MIN_WEIGHT, min(self.MAX_WEIGHT, new_weight))

    def as_dict(self) -> dict[str, float]:
        """Return a copy of the weights for serialization."""
        return dict(self.pattern_weights)

    def save_weights(self, path: str) -> None:
        """Save weights to a JSON file (atomic write).

        Args:
            path: File path to write. The write is atomic (temp file +
                rename) so a crash cannot corrupt it.
        """
        data = {"version": 1, "weights": self.as_dict()}
        directory = os.path.dirname(os.path.abspath(path))
        fd, tmp_path = tempfile.mkstemp(dir=directory, suffix=".tmp", prefix="weights_")
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(data, f)
            os.replace(tmp_path, path)
        except Exception as e:
            logger.exception(f"save weights failed: {e}")
            try:
                os.unlink(tmp_path)
            except OSError as oe:
                logger.debug(repr(oe))
            raise

    def load_weights(self, path: str) -> None:
        """Load weights from a JSON file written by :meth:`save_weights`.

        Missing or malformed files are logged and ignored — the
        existing weights are left untouched so perception degrades
        gracefully to the default (untrained) weights.
        """
        if not os.path.exists(path):
            return
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            loaded = data.get("weights", data) if isinstance(data, dict) else {}
            self.pattern_weights = {
                k: float(v)
                for k, v in loaded.items()
                if isinstance(k, str) and isinstance(v, int | float)
            }
        except (OSError, json.JSONDecodeError, ValueError):
            logger.warning("Failed to load pattern weights from %s", path, exc_info=True)


# ─── Probabilistic intent classification ──────────────────────
#
# Instead of a single deterministic intent, we compute a probability
# distribution over likely intents using a Bayesian scheme:
#
#     P(intent | text) ∝ likelihood(text | intent) × prior(intent)
#
# The likelihood is approximated by the number of pattern matches for
# each intent category, weighted by the learned pattern weights
# (perceptual learning). The prior is the context prior — what intents
# are likely given conversation history (e.g. after a greeting, a
# self-inquiry is more likely than a farewell). We use a small
# smoothing floor so unmatched intents get a non-zero (but tiny)
# probability, avoiding hard zeros that would make the distribution
# uninformative when the cognition engine wants to consider
# alternatives.

def _mean_word_vector(text: str, embeddings) -> np.ndarray | None:
    """Compute an average word-vector for a piece of text."""
    words = _SENTIMENT_WORD_RE.findall(text.lower())
    if not words:
        return None
    vectors = [embeddings.get_word_vector(w) for w in words]
    vectors = [v for v in vectors if v is not None]
    if not vectors:
        return None
    return np.mean(vectors, axis=0)


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


# Cache for intent prototype word vectors. The prototypes are static
# (defined in _INTENT_PROTOTYPES above), so their GloVe vectors never
# change. Without this cache, _compute_semantic_intent_probabilities
# recomputes ~30 _mean_word_vector calls per turn — each doing a
# dict lookup + np.mean — for data that's identical every time.
# The cache is weakly keyed on the EmbeddingStore itself: id() keys
# would leak stale entries and could be recycled by a new store after
# the old one is collected — returning vectors computed against the
# wrong embedding space.
_intent_prototype_cache: weakref.WeakKeyDictionary[
    object, dict[Intent, list[np.ndarray]]
] = weakref.WeakKeyDictionary()


def _get_prototype_vectors(embeddings) -> dict[Intent, list[np.ndarray]]:
    """Get cached prototype vectors for the given embedding store."""
    try:
        cached = _intent_prototype_cache.get(embeddings)
    except TypeError:
        cached = None
    if cached is not None:
        return cached
    vectors: dict[Intent, list[np.ndarray]] = {}
    for intent, phrases in _INTENT_PROTOTYPES.items():
        vecs = []
        for phrase in phrases:
            v = _mean_word_vector(phrase, embeddings)
            if v is not None:
                vecs.append(v)
        if vecs:
            vectors[intent] = vecs
    try:
        _intent_prototype_cache[embeddings] = vectors
    except TypeError:
        pass  # duck-typed stores that can't be weak-referenced just skip caching
    return vectors


def _compute_semantic_intent_probabilities(
    text: str, embeddings
) -> dict[Intent, float]:
    """Compute intent likelihoods from semantic similarity to prototypes."""
    if not embeddings or not embeddings.has_embeddings:
        return {}

    in_vec = _mean_word_vector(text, embeddings)
    if in_vec is None:
        return {}

    prototype_vecs = _get_prototype_vectors(embeddings)

    scores: dict[Intent, float] = {}
    for intent, p_vecs in prototype_vecs.items():
        sims = [_cosine_similarity(in_vec, v) for v in p_vecs]
        if sims:
            scores[intent] = sum(sims) / len(sims)

    # Shift into positive range and normalise to a probability distribution.
    if not scores:
        return {}
    min_score = min(scores.values())
    for k in scores:
        scores[k] = max(0.0, scores[k] - min_score)
    _normalize_probabilities(scores)
    return scores


def _blend_semantic_intent(
    pattern_probs: dict[Intent, float],
    semantic_probs: dict[Intent, float],
    pattern_weight: float = 0.65,
) -> dict[Intent, float]:
    """Blend pattern-based and embedding-based intent probabilities."""
    if not semantic_probs:
        return pattern_probs
    keys = set(pattern_probs) | set(semantic_probs)
    blended: dict[Intent, float] = {}
    for k in keys:
        p = pattern_probs.get(k, 0.0)
        s = semantic_probs.get(k, 0.0)
        blended[k] = pattern_weight * p + (1 - pattern_weight) * s
    _normalize_probabilities(blended)
    return blended


def _normalize_probabilities(probs: dict[Intent, float]) -> None:
    """Normalise a probability dict in place so values sum to 1."""
    total = sum(probs.values())
    if total <= 0:
        # Uniform fallback.
        n = len(probs) if probs else 1
        for k in probs:
            probs[k] = 1.0 / n
        return
    for k in probs:
        probs[k] = probs[k] / total


def compute_intent_probabilities(
    text: str,
    pattern_weights: PatternWeights | None = None,
    context_prior: dict[Intent, float] | None = None,
) -> dict[Intent, float]:
    """Compute a probability distribution over intents for ``text``.

    This is a Bayesian classifier. For each intent, the (unnormalised)
    posterior is::

        score(intent) = Σ (match_count × pattern_weight) × prior(intent)

    where the sum is over pattern categories that map to that intent,
    ``match_count`` is how many of the category's patterns matched the
    text, ``pattern_weight`` is the learned reliability of that
    category (perceptual learning, default 1.0), and ``prior(intent)``
    is the context prior (default uniform).

    The result is normalised to a proper probability distribution. A
    small smoothing floor is added so every intent that has any
    pattern category gets a non-zero probability, even when no
    patterns match — this keeps the distribution useful for the
    cognition engine when the input is ambiguous.

    Args:
        text: The (lowercased) input text.
        pattern_weights: Optional learned pattern weights. If ``None``,
            all categories use the default weight of 1.0.
        context_prior: Optional prior over intents reflecting
            conversation history. If ``None``, a uniform prior is
            used. Intents not present in the prior get a small floor.

    Returns:
        A mapping from :class:`Intent` to probability. The highest
        probability intent corresponds to the most likely
        classification; ties or near-ties signal ambiguity the
        cognition engine can resolve with context.
    """
    lower = text.lower()
    has_question = "?" in lower or _detect_question_type(lower) != QuestionType.NONE

    # Aggregate weighted match scores per intent.
    scores = _score_intent_patterns(lower, has_question, pattern_weights)

    # Build the full distribution over all intents, applying the prior.
    probs = _apply_prior_and_normalize(scores, 0.01, context_prior)
    return probs


def _score_intent_patterns(
    lower: str,
    has_question: bool,
    pattern_weights: PatternWeights | None,
) -> dict[Intent, float]:
    """Aggregate weighted match scores per intent from pattern categories.

    For each pattern category, counts matches against the text, applies
    the learned pattern weight, and resolves context-sensitive intent
    variants (e.g. GREETING vs GREETING_QUESTION).
    """
    scores: dict[Intent, float] = {}
    for name, (patterns, base_intent) in _PATTERN_CATEGORIES.items():
        count = _count_matches(patterns, lower)
        if count == 0:
            continue
        weight = pattern_weights.get(name) if pattern_weights else 1.0
        # Resolve context-sensitive variants.
        if base_intent == Intent.GREETING:
            intent = Intent.GREETING_QUESTION if has_question else Intent.GREETING
        elif base_intent == Intent.PHILOSOPHY:
            intent = Intent.PHILOSOPHY if has_question else Intent.REFLECTION
        else:
            intent = base_intent
        scores[intent] = scores.get(intent, 0.0) + count * weight
    return scores


def _apply_prior_and_normalize(
    scores: dict[Intent, float],
    smoothing: float,
    context_prior: dict[Intent, float] | None,
) -> dict[Intent, float]:
    """Build the full intent distribution, apply the prior, normalize, and prune.

    Combines per-intent likelihood scores with the context prior to
    produce a proper probability distribution. Negligible entries are
    pruned (but the top intent is always kept), then re-normalized.
    """
    # Build the full distribution over all intents, applying the prior.
    all_intents = list(Intent)
    prior = context_prior or {}
    probs: dict[Intent, float] = {}
    for intent in all_intents:
        likelihood = scores.get(intent, 0.0) + smoothing
        # Context prior: use provided value, or a uniform floor.
        p = prior.get(intent, 1.0 / len(all_intents) if not prior else 0.01)
        probs[intent] = likelihood * max(p, 1e-4)

    _normalize_probabilities(probs)

    # Drop negligible entries to keep the dict small, but always keep
    # the top intent and any intent with meaningful mass.
    if probs:
        top = max(probs, key=lambda k: probs[k])
        probs = {k: round(v, 6) for k, v in probs.items() if v >= 0.001 or k == top}
        # Re-normalise after pruning so the distribution still sums to 1.
        _normalize_probabilities(probs)
    return probs


# ─── Multisensory integration framework ───────────────────────
#
# Multisensory integration is the brain's combination of information
# from different senses into a unified percept. The superior
# colliculus and cortical "convergence zones" (Stein & Stanford, 2008)
# receive inputs from multiple modalities; when two modalities agree,
# cross-modal binding boosts confidence (the "reverse hierarchy"
# effect); when they disagree, the conflict is flagged for resolution.
#
# Currently only the TEXT modality is active, but the framework is
# designed for future expansion to AUDIO, VISUAL, and TEMPORAL
# (time-based pattern) modalities. The integrator performs cross-modal
# binding and conflict detection so that adding new modalities later
# requires only providing inputs, not changing the integration logic.
#
# Reference:
# - Stein, B. E., & Stanford, T. R. (2008). Multisensory integration:
#   current issues from the perspective of the single neuron.
#   *Nature Reviews Neuroscience*, 9(4), 255–266.


class SensoryModality(Enum):
    """A sensory channel that can provide input to perception."""

    TEXT = "text"
    AUDIO = "audio"  # future: prosody, speech features
    VISUAL = "visual"  # future: gestures, facial expressions
    TEMPORAL = "temporal"  # time-based patterns (pauses, rhythm)


@dataclass(slots=True)
class MultisensoryInput:
    """A single input from one sensory modality.

    Attributes:
        modality: Which sensory channel this input came from.
        content: The raw content (text for TEXT, feature vector for
            future modalities).
        confidence: How reliable this modality considers its own
            reading, in ``[0, 1]``.
        intent_hint: Optional intent suggested by this modality
            (TEXT uses the rule-based classifier; future modalities
            may infer intent from prosody or gesture).
        timestamp: Optional ordering key for temporal fusion.
    """

    modality: SensoryModality
    content: str
    confidence: float = 1.0
    intent_hint: Intent | None = None
    timestamp: float = 0.0


@dataclass(slots=True)
class IntegratedPerception(Perception):
    """A perception that fuses multiple sensory modalities.

    Extends :class:`Perception` with cross-modal confidence and
    conflict detection. When all active modalities agree, the
    cross-modal confidence is higher than any single modality's
    (binding gain). When they disagree, a conflict is flagged and the
    confidence is reduced.
    """

    cross_modal_confidence: float = 0.0
    modalities_present: list[str] = field(default_factory=list)
    modality_intents: dict[str, Intent] = field(default_factory=dict)
    conflict_detected: bool = False
    conflict_description: str = ""


class MultisensoryIntegrator:
    """Fuses inputs from multiple sensory modalities into one percept.

    Cross-modal binding: when two modalities suggest the same intent,
    confidence increases (the brain's multisensory convergence zones
    amplify consistent signals). Sensory conflict: when modalities
    disagree, the conflict is flagged and confidence is reduced so the
    cognition engine can request clarification or fall back to the
    most reliable modality.

    Currently only :attr:`SensoryModality.TEXT` is active, but the
    framework supports AUDIO, VISUAL, and TEMPORAL modalities for
    future expansion — adding a modality only requires producing
    :class:`MultisensoryInput` instances; the integration logic
    stays the same.
    """

    # Modalities currently producing real input. Others are accepted
    # but flagged as inactive so the framework is ready for expansion.
    ACTIVE_MODALITIES: frozenset[SensoryModality] = frozenset({SensoryModality.TEXT})

    def __init__(self, *, embeddings=None, network=None) -> None:
        """Initialize the multisensory integrator.

        Args:
            embeddings: Optional word-embedding model used for semantic
                intent classification.
            network: Optional concept network used for multi-word topic
                merging.
        """
        self._embeddings = embeddings
        self._network = network

    def integrate(
        self,
        inputs: list[MultisensoryInput],
        pattern_weights: PatternWeights | None = None,
        context_prior: dict[Intent, float] | None = None,
    ) -> IntegratedPerception:
        """Fuse multiple modality inputs into a single integrated percept.

        Args:
            inputs: One or more :class:`MultisensoryInput` instances,
                potentially from different modalities.
            pattern_weights: Optional learned pattern weights for the
                text-modality classifier.
            context_prior: Optional conversation-history prior.

        Returns:
            An :class:`IntegratedPerception` whose base fields come
            from the (currently sole active) TEXT modality, augmented
            with cross-modal confidence, the set of modalities
            present, per-modality intent hints, and a conflict flag.
        """
        if not inputs:
            # Degenerate case: no input at all.
            return _empty_integrated_perception()

        # Separate active vs inactive (future) modalities.
        active = [i for i in inputs if i.modality in self.ACTIVE_MODALITIES]
        inactive = [i for i in inputs if i.modality not in self.ACTIVE_MODALITIES]
        modalities_present = [i.modality.value for i in inputs]

        base = self._integrate_base_perception(inputs, active, pattern_weights, context_prior)

        # Collect per-modality intent hints.
        modality_intents: dict[str, Intent] = {}
        for inp in inputs:
            if inp.intent_hint is not None:
                modality_intents[inp.modality.value] = inp.intent_hint
            elif inp.modality == SensoryModality.TEXT:
                modality_intents[inp.modality.value] = base.intent

        cross_modal_confidence, conflict_detected, conflict_description = (
            _cross_modal_binding(modality_intents, base.confidence)
        )

        if inactive:
            inactive_names = ", ".join(i.modality.value for i in inactive)
            if conflict_description:
                conflict_description += f" (inactive modalities ignored: {inactive_names})"
            else:
                conflict_description = f"Inactive modalities ignored: {inactive_names}"

        # Promote the base Perception to an IntegratedPerception.
        return _promote_to_integrated(
            base,
            cross_modal_confidence,
            modalities_present,
            modality_intents,
            conflict_detected,
            conflict_description,
        )

    def _integrate_base_perception(
        self,
        inputs: list[MultisensoryInput],
        active: list[MultisensoryInput],
        pattern_weights: PatternWeights | None,
        context_prior: dict[Intent, float] | None,
    ) -> Perception:
        """Run the text classifier on the primary (or fallback) input."""
        # Run the text classifier on the primary TEXT input.
        text_input = next((i for i in active if i.modality == SensoryModality.TEXT), None)
        if text_input is not None:
            return perceive(
                text_input.content,
                embeddings=self._embeddings,
                network=self._network,
                pattern_weights=pattern_weights,
                context_prior=context_prior,
            )
        # No active text — build a minimal perception from the
        # first available input so the framework still functions.
        fallback = inputs[0]
        return perceive(
            fallback.content,
            embeddings=self._embeddings,
            network=self._network,
            pattern_weights=pattern_weights,
            context_prior=context_prior,
        )


def _empty_integrated_perception() -> IntegratedPerception:
    """Build the degenerate IntegratedPerception for the no-input case."""
    return IntegratedPerception(
        raw_text="",
        intent=Intent.UNKNOWN,
        question_type=QuestionType.NONE,
        topics=[],
        sentiment=0.0,
        sentiment_label="neutral",
        emotion_word="",
        is_about_genesis=False,
        is_about_user=False,
        is_about_code=False,
        is_about_emotion=False,
        is_about_existence=False,
        entities={},
        key_phrases=[],
        word_count=0,
        confidence=0.0,
        intent_probabilities={},
        cross_modal_confidence=0.0,
        modalities_present=[],
        modality_intents={},
        conflict_detected=False,
        conflict_description="No sensory input provided.",
    )


def _cross_modal_binding(
    modality_intents: dict[str, Intent], base_confidence: float
) -> tuple[float, bool, str]:
    """Cross-modal binding & conflict detection.

    Returns (cross_modal_confidence, conflict_detected, conflict_description).
    """
    hinted_intents = list(modality_intents.values())
    unique_intents = set(hinted_intents)
    conflict_detected = len(unique_intents) > 1
    conflict_description = ""

    if conflict_detected:
        counts = Counter(h.name for h in hinted_intents)
        conflict_description = "Modality conflict: " + ", ".join(
            f"{k}×{v}" for k, v in counts.items()
        )
        # Reduce confidence: disagreement lowers reliability.
        cross_modal_confidence = base_confidence * 0.6
    elif len(hinted_intents) >= 2:
        # Binding gain: agreement across modalities boosts
        # confidence (Stein & Stanford, 2008 — superadditive
        # enhancement for congruent multisensory cues).
        cross_modal_confidence = min(base_confidence * 1.2, 1.0)
    else:
        cross_modal_confidence = base_confidence
    return cross_modal_confidence, conflict_detected, conflict_description


def _promote_to_integrated(
    base: Perception,
    cross_modal_confidence: float,
    modalities_present: list[str],
    modality_intents: dict[str, Intent],
    conflict_detected: bool,
    conflict_description: str,
) -> IntegratedPerception:
    """Promote a base Perception to an IntegratedPerception."""
    return IntegratedPerception(
        raw_text=base.raw_text,
        intent=base.intent,
        question_type=base.question_type,
        topics=base.topics,
        sentiment=base.sentiment,
        sentiment_label=base.sentiment_label,
        emotion_word=base.emotion_word,
        is_about_genesis=base.is_about_genesis,
        is_about_user=base.is_about_user,
        is_about_code=base.is_about_code,
        is_about_emotion=base.is_about_emotion,
        is_about_existence=base.is_about_existence,
        entities=base.entities,
        key_phrases=base.key_phrases,
        word_count=base.word_count,
        confidence=base.confidence,
        intent_probabilities=base.intent_probabilities,
        cross_modal_confidence=round(cross_modal_confidence, 4),
        modalities_present=modalities_present,
        modality_intents=modality_intents,
        conflict_detected=conflict_detected,
        conflict_description=conflict_description,
    )
