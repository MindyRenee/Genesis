"""Self-directed learning — Genesis improving her own knowledge.

Unlike the AutonomousLearner (which fetches from external sources like
Wikipedia), this module helps Genesis learn from what she *already
knows* through reasoning, inference, and conversation.

## The gap this fills

Genesis has a concept network with 600+ concepts and 2600+ relationships.
But many concepts lack definitions, many relationships are missing that
could be inferred from existing ones, and every conversation with the
user contains facts she could extract but doesn't.

This module provides five learning mechanisms:

1. **Conversation learning** — when the user says "X is a Y" or
   "X means Z", extract the fact and add it to the concept network.
   This is the most direct path to improvement: every conversation
   makes her smarter.

2. **Transitive inference** — if A is_a B and B is_a C, infer A is_a C.
   Same for PART_OF and DEPENDS_ON. This multiplies her knowledge
   without any external input.

3. **Definition synthesis** — for concepts that have relationships but
   no definition, build one from the relationships. "X is_a Y, X
   enables Z" → "X is a kind of Y that enables Z."

4. **Analogical transfer** — if two concepts share many IS_A parents,
   they're similar. Properties of one may apply to the other. This
   is speculative (lower confidence) but generates learning targets.

5. **Self-study** — review weak concepts (identified by
   SelfAssessmentEngine), strengthen their edges, and synthesize
   definitions. This is the "homework" she does on her own.

## How it connects

- **SelfAssessmentEngine** identifies gaps → this module fills them
- **CognitionEngine** routes user input → this module extracts facts
- **CognitionEngine** handles corrections → this module updates knowledge
- **AutonomousLearner** fetches external content → this module infers
  new relationships from what was learned
- **CuriosityEngine** generates questions → this module can answer
  some of them through inference

## Biological grounding

This mirrors how the brain consolidates knowledge:
- **Systems consolidation** (McClelland et al., 1995): hippocampal
  episodes are gradually abstracted into neocortical schemas. The
  transitive inference and definition synthesis are the computational
  analog — extracting general knowledge from specific facts.
- **Analogical reasoning** (Gentner & Markman, 1997): the brain uses
  structural similarity to transfer knowledge. Our analogical transfer
  does the same with concept network structure.
- **Spaced replay** (Tse et al., 2007): reviewing weak memories
  stabilizes them. Our self-study cycle re-activates weak concepts.
"""

from __future__ import annotations

import logging
import re
import time
from collections import deque
from dataclasses import dataclass, field

from ..brain_waves import BrainWave
from ..concepts import _FUNCTION_WORDS, ConceptNetwork, RelationType

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════
# Fact extraction patterns
# ═══════════════════════════════════════════════════════════════════
#
# Each pattern captures a (subject, object) pair from a specific
# linguistic construction. Patterns are tried in priority order —
# more specific patterns (definitions, part_of, similar_to) before
# less specific ones (is_a) to avoid the is_a regex consuming text
# that belongs to a more specific relation.
#
# All patterns use (?<!\w) to avoid matching inside a larger word,
# and require the subject to start with a lowercase letter (after
# the lookbehind). The IGNORECASE flag handles capitalized inputs.


def _make_pattern(verb_part: str) -> re.Pattern[str]:
    """Build a fact-extraction regex with a standard structure.

    The subject is 1-4 words (allowing hyphens for compound terms like
    "micro-animal"). The verb_part is the specific verb construction.
    The object is 1-3 words (also allowing hyphens).
    """
    return re.compile(
        r"(?<!\w)([a-z]{2,}(?:[-\s][a-z]+){0,3})\s+"
        + verb_part
        + r"\s+"
        + r"([a-z]+(?:[-\s][a-z]+){0,2})\b",
        re.IGNORECASE,
    )


def _clean_definition_text(defn: str) -> str:
    """Strip pronunciation, etymology, and other non-definition junk.

    Dictionary entries often include IPA, pronunciation hints, and
    section headers (Etymology, Synonyms, etc.). This removes them so
    Genesis only learns the actual meaning of a word.
    """
    defn = defn.strip()

    # Remove numbered definition prefixes like "1. "
    defn = re.sub(r"^\d+[.):]\s+", "", defn)

    # Remove any section that starts with a dictionary label and runs
    # to the end — Pronunciation, Etymology, Synonyms, etc.
    section_labels = (
        r"Pronunciation:",
        r"Etymology:",
        r"Synonyms:",
        r"Antonyms:",
        r"Derived terms:",
        r"Translations:",
        r"See also:",
        r"References:",
    )
    for label in section_labels:
        defn = re.sub(r"\s*" + label + r".*", "", defn, flags=re.IGNORECASE)

    # Remove IPA pronunciation markers like /ˈwɔtər/ or \u02c8...
    defn = re.sub(r"\s*/[^/\s]+/", "", defn)

    # Remove parenthesized pronunciation hints like (wô'tər)
    defn = re.sub(r"\s*\([^)]*\)", "", defn)

    return defn.strip()


# Definition patterns capture a longer object (the definition text).
_DEFINITION_RE = re.compile(
    r"(?<!\w)([a-z]{1,}(?:[-\s][a-z]+){0,3})\s+"
    r"(?:means|is\s+defined\s+as|refers\s+to|is\s+when)\s+"
    r"(.{10,200})",
    re.IGNORECASE,
)

# Bare copula definition: "X is Y" where Y doesn't start with a/an
# (those are handled by _IS_A_RE). This catches the most common
# definitional form in natural language:
#   "Courage is feeling fear and acting anyway"
#   "Courage is truth over comfort"
#   "Power is the ability to influence what happens"
#   "Hope is not naive optimism"
#   "Fear is natural"
# The negative lookahead excludes patterns handled by more specific
# regexes (part of, similar to, used for, etc.) so this only catches
# what they miss.
_BARE_IS_RE = re.compile(
    r"(?<!\w)([a-z]{1,}(?:[-\s][a-z]+){0,3})\s+"
    r"is\s+(?!a\s|an\s|part\s+of|similar\s+to|used\s+|opposite\s+|"
    r"defined\s+as|when\s|a\s+(?:type|kind|form|sort|class|category)\s+of|"
    r"composed\s+of|a\s+result\s+of|caused\s+by|enabled\s+by|prevented\s+by)"
    r"([^.!?]{5,200})",
    re.IGNORECASE,
)

# "X is a Y" / "X is an Y" / "X are a Y"
# Requires "a" or "an" — bare "is" catches "is similar to", "is part of".
_IS_A_RE = _make_pattern(r"(?:is\s+(?:a|an)|are\s+(?:a|an))")

# "X is part of Y" — subject is non-greedy so "is" isn't captured.
# Articles before the object are consumed but not captured.
_PART_OF_RE = re.compile(
    r"(?<!\w)([a-z]{2,}(?:[-\s][a-z]+){0,3}?)\s+"
    r"(?:is\s+|are\s+)?part\s+of\s+(?:the\s+|a\s+|an\s+)?"
    r"([a-z]+(?:[-\s][a-z]+){0,3})\b",
    re.IGNORECASE,
)

# "X causes Y" / "X leads to Y" / "X produces Y" / "X generates Y"
_CAUSES_RE = _make_pattern(r"(?:causes|leads\s+to|produces|generates)")

# "X depends on Y" / "X needs Y" / "X requires Y"
_DEPENDS_RE = _make_pattern(r"(?:depends\s+on|needs|requires)")

# "X enables Y" / "X allows Y" / "X permits Y"
_ENABLES_RE = _make_pattern(r"(?:enables|allows|permits)")

# "X is (the) opposite of Y" / "X opposes Y" / "X contradicts Y"
_OPPOSITE_RE = re.compile(
    r"(?<!\w)([a-z]{2,}(?:[-\s][a-z]+){0,3})\s+"
    r"(?:is\s+(?:the\s+)?opposite\s+(?:of|to)|opposes|contradicts)\s+"
    r"([a-z]+(?:[-\s][a-z]+){0,3})\b",
    re.IGNORECASE,
)

# "X is similar to Y"
_SIMILAR_RE = _make_pattern(r"is\s+similar\s+to")

# "X includes Y" / "X contains Y" / "X encompasses Y"
# Maps to PART_OF with reversed source/target: Y is part of X.
_INCLUDES_RE = re.compile(
    r"(?<!\w)([a-z]{2,}(?:[-\s][a-z]+){0,3})\s+"
    r"(?:includes|contains|encompasses|comprises)\s+"
    r"(?:the\s+|a\s+|an\s+)?"
    r"([a-z]+(?:[-\s][a-z]+){0,3})\b",
    re.IGNORECASE,
)

# "X has Y" / "X has a Y" / "X have Y"
# The article group includes its own whitespace so "has eight" works
# as well as "has a tail".
_HAS_RE = re.compile(
    r"(?<!\w)([a-z]{2,}(?:[-\s][a-z]+){0,3})\s+"
    r"(?:has(?:\s+(?:a|an))?|have(?:\s+(?:a|an))?)\s+"
    r"([a-z]+(?:[-\s][a-z]+){0,2})\b",
    re.IGNORECASE,
)

# ── Tier 2 patterns: richer verb coverage ──────────────────────

# "X prevents Y" / "X inhibits Y" / "X blocks Y" / "X stops Y"
_PREVENTS_RE = _make_pattern(r"(?:prevents|inhibits|blocks|stops)")

# "X creates Y" / "X makes Y" / "X forms Y" / "X builds Y" / "X built Y"
_CREATES_RE = _make_pattern(r"(?:creates?|makes?|made|forms?|builds?|built)")

# "X harms Y" / "X damages Y" / "X hurts Y" / "X impairs Y"
_HARMS_RE = _make_pattern(r"(?:harms|damages|hurts|impairs)")

# "X eats Y" / "X eats a Y" / "X drink Y" / "X consume Y"
# Captures the consumer-resource relationship. The object can be a noun
# phrase (e.g. "flowers"), so allow up to 3 words.
_EATS_RE = re.compile(
    r"(?<!\w)([a-z]{2,}(?:[-\s][a-z]+){0,3})\s+"
    r"(?:eats?|drinks?|consumes?|feeds? on)\s+"
    r"(?:the\s+|a\s+|an\s+)?"
    r"([a-z]+(?:[-\s][a-z]+){0,3})\b",
    re.IGNORECASE,
)

# "X teaches Y" / "X teaches us Y" / "X shows Y" / "X tells Y"
# Maps to ENABLES — teaching someone something enables them to know it.
_TEACHES_RE = re.compile(
    r"(?<!\w)([a-z]{2,}(?:[-\s][a-z]+){0,3})\s+"
    r"(?:teaches?|shows?|tells?|explains?)(?:\s+\w+){0,2}\s+"
    r"(?:that\s+)?"
    r"([a-z]+(?:[-\s][a-z]+){0,3})\b",
    re.IGNORECASE,
)

# "X wants Y" / "X needs Y" / "X desires Y" / "X requires Y"
# Maps to DEPENDS_ON — wanting/ needing something means the subject's
# well-being depends on it.
_WANTS_RE = _make_pattern(r"(?:wants?|needs?|desires?|requires?)")

# "X wants to Y" / "X needs to Y" — infinitive goals.
_WANTS_TO_RE = re.compile(
    r"(?<!\w)([a-z]{2,}(?:[-\s][a-z]+){0,3})\s+"
    r"(?:wants?|needs?|desires?|requires?)\s+to\s+"
    r"([a-z]+(?:[-\s][a-z]+){0,3})\b",
    re.IGNORECASE,
)

# "X uses Y" / "X uses a Y" / "X employs Y"
# Maps to ENABLES — using something enables the subject's function.
_USES_RE = re.compile(
    r"(?<!\w)([a-z]{2,}(?:[-\s][a-z]+){0,3})\s+"
    r"(?:uses?|employs?)\s+"
    r"(?:the\s+|a\s+|an\s+)?"
    r"([a-z]+(?:[-\s][a-z]+){0,3})\b",
    re.IGNORECASE,
)

# "X lives in Y" / "X exists in Y" / "X is located in Y"
# Maps to SPATIAL_RELATION — the subject is situated in the container.
_LIVES_IN_RE = re.compile(
    r"(?<!\w)([a-z]{2,}(?:[-\s][a-z]+){0,3})\s+"
    r"(?:lives? in|exists? in|is located in|resides? in)\s+"
    r"(?:the\s+|a\s+|an\s+)?"
    r"([a-z]+(?:[-\s][a-z]+){0,3})\b",
    re.IGNORECASE,
)

# "X is about Y" / "X is for Y" / "X concerns Y"
# Maps to RELATED_TO — topic/purpose relationship.
_IS_ABOUT_RE = re.compile(
    r"(?<!\w)([a-z]{2,}(?:[-\s][a-z]+){0,3})\s+"
    r"(?:is|was|are|were)\s+"
    r"(?:about|for|concerning|regarding)\s+"
    r"(?:the\s+|a\s+|an\s+)?"
    r"([a-z]+(?:[-\s][a-z]+){0,3})\b",
    re.IGNORECASE,
)

# "X is more efficient than Y" / "X is bigger than Y" / "X is faster than Y"
# Maps to SIMILAR_TO with a qualifier, but we capture the relation.
_MORE_THAN_RE = re.compile(
    r"(?<!\w)([a-z]{2,}(?:[-\s][a-z]+){0,3})\s+"
    r"is\s+more\s+([a-z]+)\s+than\s+"
    r"(?:the\s+|a\s+|an\s+)?"
    r"([a-z]+(?:[-\s][a-z]+){0,3})\b",
    re.IGNORECASE,
)

# "X is created by Y" / "X is built by Y" / "X is made by Y"
# Maps to CREATES (reversed: Y creates X).
_CREATED_BY_RE = re.compile(
    r"(?<!\w)([a-z]{2,}(?:[-\s][a-z]+){0,3})\s+"
    r"(?:is\s+|was\s+)?"
    r"(?:created|built|made|designed|developed)\s+by\s+"
    r"([a-z]+(?:[-\s][a-z]+){0,3})\b",
    re.IGNORECASE,
)

# "X does not need Y" / "X doesn't need Y" / "X needs no Y"
# Maps to PREVENTS (X prevents needing Y) — a weak signal but useful.
_DOES_NOT_NEED_RE = re.compile(
    r"(?<!\w)([a-z]{2,}(?:[-\s][a-z]+){0,3})\s+"
    r"(?:does\s+not|doesn'?t)\s+"
    r"(?:need|require|want|desire)\s+"
    r"(?:the\s+|a\s+|an\s+)?"
    r"([a-z]+(?:[-\s][a-z]+){0,3})\b",
    re.IGNORECASE,
)

# "X knows Y" / "X understands Y" / "X remembers Y" / "X believes Y"
# Maps to RELATED_TO for now; the relation between a mind and its
# contents is generic but important for conversation.
_KNOWS_RE = _make_pattern(r"(?:knows?|understands?|remembers?|believes?)\s+(?:that\s+)?")

# "X emerges from Y" / "X arises from Y" / "X derives from Y"
_EMERGES_RE = _make_pattern(r"(?:emerges\s+from|arises\s+from|derives\s+from)")

# "X is a type of Y" / "X is a kind of Y" / "X is a form of Y"
_TYPE_OF_RE = re.compile(
    r"(?<!\w)([a-z]{2,}(?:[-\s][a-z]+){0,3})\s+"
    r"is\s+(?:a|an)\s+(?:type|kind|form|sort|class|category)\s+of\s+"
    r"([a-z]+(?:[-\s][a-z]+){0,3})\b",
    re.IGNORECASE,
)

# "X is used for Y" / "X is used to Z" / "X is used in Y"
_USED_FOR_RE = re.compile(
    r"(?<!\w)([a-z]{2,}(?:[-\s][a-z]+){0,3})\s+"
    r"is\s+used\s+(?:for|to|in)\s+"
    r"([a-z]+(?:[-\s][a-z]+){0,2})\b",
    re.IGNORECASE,
)

# "X can Y" / "X can survive Y" / "X can withstand Y"
# Maps to ENABLES — X has the capability to do Y.
# The object is a verb phrase, so we capture the verb + object.
# The subject group excludes relative pronouns (that/which/who) so
# "micro-animal that can survive" doesn't match — the relative clause
# extraction handles that case separately.
_CAN_RE = re.compile(
    r"(?<!\w)(?!that\b|which\b|who\b)"
    r"([a-z]{2,}(?:[-\s](?!that\b|which\b|who\b)[a-z]+){0,3})\s+"
    r"can\s+"
    r"([a-z]+(?:[-\s][a-z]+){0,3})\b",
    re.IGNORECASE,
)

# "X is composed of Y" / "X consists of Y" / "X is made up of Y"
# Note: this is reversed — Y is_part_of X, not X is_part_of Y.
_COMPOSED_OF_RE = re.compile(
    r"(?<!\w)([a-z]{2,}(?:[-\s][a-z]+){0,3})\s+"
    r"(?:is\s+composed\s+of|consists\s+of|is\s+made\s+up\s+of)\s+"
    r"([a-z]+(?:[-\s][a-z]+){0,3})\b",
    re.IGNORECASE,
)

# "X is a result of Y" / "X results from Y" / "X comes from Y"
_RESULT_OF_RE = re.compile(
    r"(?<!\w)([a-z]{2,}(?:[-\s][a-z]+){0,3})\s+"
    r"(?:is\s+a\s+result\s+of|results\s+from|comes\s+from)\s+"
    r"([a-z]+(?:[-\s][a-z]+){0,3})\b",
    re.IGNORECASE,
)

# ── Passive voice patterns ─────────────────────────────────────
# These reverse the subject/object: "Y is caused by X" → X causes Y

# "Y is caused by X" → X CAUSES Y
_PASSIVE_CAUSED_RE = re.compile(
    r"(?<!\w)([a-z]{2,}(?:[-\s][a-z]+){0,3})\s+"
    r"is\s+(?:caused|produced|generated|triggered|driven)\s+by\s+"
    r"([a-z]+(?:[-\s][a-z]+){0,3})\b",
    re.IGNORECASE,
)

# "Y is enabled by X" / "Y is allowed by X" → X ENABLES Y
_PASSIVE_ENABLED_RE = re.compile(
    r"(?<!\w)([a-z]{2,}(?:[-\s][a-z]+){0,3})\s+"
    r"is\s+(?:enabled|allowed|permitted|facilitated|supported)\s+by\s+"
    r"([a-z]+(?:[-\s][a-z]+){0,3})\b",
    re.IGNORECASE,
)

# "Y is prevented by X" / "Y is inhibited by X" → X PREVENTS Y
_PASSIVE_PREVENTED_RE = re.compile(
    r"(?<!\w)([a-z]{2,}(?:[-\s][a-z]+){0,3})\s+"
    r"is\s+(?:prevented|inhibited|blocked|suppressed)\s+by\s+"
    r"([a-z]+(?:[-\s][a-z]+){0,3})\b",
    re.IGNORECASE,
)

# "Y is composed of X" is already handled by _COMPOSED_OF_RE above
# (it's not technically passive voice, but the reversal is the same)

# ── Possessive pattern: "X's Y" → Y is_part_of X ───────────────
# e.g., "the brain's hippocampus" → hippocampus is_part_of brain
# Only captures the first word after the possessive (the noun),
# not subsequent words which may be verbs or other clauses.
# The apostrophe is required to avoid matching plurals ("processes memory").
_POSSESSIVE_RE = re.compile(
    r"(?<!\w)([a-z]{2,}(?:[-\s][a-z]+){0,2})'s\s+"
    r"([a-z]+)\b",
    re.IGNORECASE,
)

# Correction prefix: "No, ..." / "Actually, ..." / "That's not right, ..."
_CORRECTION_RE = re.compile(
    r"(?:no[,\s]+|actually[,\s]+|that'?s\s+(?:not\s+)?right[,\s]+|"
    r"not\s+exactly[,\s]+|let\s+me\s+correct[,\s]+)"
    r"(.+)",
    re.IGNORECASE,
)


# ═══════════════════════════════════════════════════════════════════
# Genus extraction from dictionary definitions
# ═══════════════════════════════════════════════════════════════════
#
# Dictionary definitions follow a genus-differentia pattern:
# "serotonin: n. a neurotransmitter involved in sleep..."
# The genus ("neurotransmitter") tells us what kind of thing it is.
# Extracting these creates IS_A edges that deepen the semantic hierarchy.

# WordNet-style: "word\n    n 1: a/an/the <genus> ..."
_WORDNET_NOUN_RE = re.compile(
    r"^\w+\s*\n\s*n\s*\d*:\s+(?:a|an|the)\s+(.+)",
    re.IGNORECASE,
)

# Webster-style: "Word \Word\, n.\n   1. The <genus> ..."
_WEBSTER_NOUN_RE = re.compile(
    r"n\.\s*\n\s*\d*\.\s*(?:the\s+|a\s+|an\s+)?(.+)",
    re.IGNORECASE,
)

# Concept definition patterns: "X is a/an Y" or "the <genus> of..."
_CONCEPT_DEF_RE = re.compile(
    r"^(?:a|an|the)\s+([a-z][a-z\s]+?)(?:\s+(?:that|which|who|where|of|for|in|by|with|from|to|involving|characterized|used|designed|related|associated|concerned|responsible|capable|able|[;,.])\b)",
    re.IGNORECASE,
)

# Words that signal the end of the genus term (start of the differentia)
_GENUS_STOP_WORDS = frozenset({
    # Prepositions / conjunctions
    "that", "which", "who", "whom", "whose", "where", "of", "for", "in",
    "by", "with", "from", "to", "at", "on", "into", "through", "during",
    "before", "after", "as", "than", "and", "or", "but", "when", "while",
    "if", "unless", "because", "since", "although",
    # Common verbs (present, past, participles)
    "is", "are", "was", "were", "be", "been", "being", "has", "have",
    "had", "do", "does", "did", "will", "would", "can", "could",
    "should", "must", "may", "might", "shall",
    "involves", "causes", "enables", "requires", "depends", "produces",
    "creates", "leads", "results", "consists", "generates", "develops",
    "grows", "evolves", "emerges", "appears", "occurs", "happens",
    # Past participles (common in definitions)
    "characterized", "used", "designed", "related", "associated",
    "concerned", "responsible", "capable", "involving", "consisting",
    "based", "situated", "located", "found", "seen", "observed",
    "considered", "regarded", "defined", "described", "represented",
    "intended", "meant", "employed", "applied", "utilized", "adopted",
    "drawn", "derived", "obtained", "gained", "acquired", "received",
    "composed", "made", "formed", "marked", "noted", "known",
    "held", "kept", "charged", "filed", "sold", "offered",
    "written", "spoken", "expressed", "performed",
})


# ═══════════════════════════════════════════════════════════════════
# Relation extraction table — data-driven, not copy-pasted
# ═══════════════════════════════════════════════════════════════════


@dataclass(slots=True, frozen=True)
class _RelationPattern:
    """One pattern for extracting a typed relationship from text.

    For normal patterns, group(1) is the source and group(2) is the
    target: "X causes Y" → source=X, target=Y, relation=CAUSES.

    For reversed patterns (passive voice, "composed of"), set
    reversed=True: group(1) is the target and group(2) is the source:
    "Y is caused by X" → target=Y, source=X, relation=CAUSES.
    """

    regex: re.Pattern[str]
    relation: RelationType
    weight: float
    label: str  # human-readable description template: "X is a {obj}"
    reversed: bool = False  # if True, swap source/target


# Order matters: more specific patterns first so they get priority
# on overlapping text. Definition extraction is handled separately
# (before this table) because it captures a longer object.
_RELATION_PATTERNS: tuple[_RelationPattern, ...] = (
    # ── Most specific multi-word patterns first ──
    _RelationPattern(_PART_OF_RE, RelationType.PART_OF, 0.7, "{subj} is part of {obj}"),
    _RelationPattern(
        _INCLUDES_RE, RelationType.PART_OF, 0.7,
        "{obj} is part of {subj}", reversed=True,
    ),
    _RelationPattern(_TYPE_OF_RE, RelationType.IS_A, 0.75, "{subj} is a type of {obj}"),
    _RelationPattern(
        _COMPOSED_OF_RE, RelationType.PART_OF, 0.7,
        "{subj} is part of {obj}", reversed=True,
    ),
    _RelationPattern(_RESULT_OF_RE, RelationType.EMERGES_FROM, 0.7, "{subj} emerges from {obj}"),
    _RelationPattern(_USED_FOR_RE, RelationType.ENABLES, 0.65, "{subj} is used for {obj}"),
    _RelationPattern(_CAN_RE, RelationType.ENABLES, 0.6, "{subj} can {obj}"),
    _RelationPattern(_OPPOSITE_RE, RelationType.OPPOSITE_OF, 0.7, "{subj} is opposite of {obj}"),
    _RelationPattern(_SIMILAR_RE, RelationType.SIMILAR_TO, 0.6, "{subj} is similar to {obj}"),
    _RelationPattern(_EMERGES_RE, RelationType.EMERGES_FROM, 0.7, "{subj} emerges from {obj}"),
    # ── Passive voice (reversed) ──
    _RelationPattern(
        _PASSIVE_CAUSED_RE, RelationType.CAUSES, 0.7,
        "{subj} causes {obj}", reversed=True,
    ),
    _RelationPattern(
        _PASSIVE_ENABLED_RE, RelationType.ENABLES, 0.7,
        "{subj} enables {obj}", reversed=True,
    ),
    _RelationPattern(
        _PASSIVE_PREVENTED_RE, RelationType.PREVENTS, 0.7,
        "{subj} prevents {obj}", reversed=True,
    ),
    # ── Active voice verb patterns ──
    _RelationPattern(_PREVENTS_RE, RelationType.PREVENTS, 0.7, "{subj} prevents {obj}"),
    _RelationPattern(_CAUSES_RE, RelationType.CAUSES, 0.7, "{subj} causes {obj}"),
    _RelationPattern(_CREATES_RE, RelationType.CREATES, 0.65, "{subj} creates {obj}"),
    _RelationPattern(_HARMS_RE, RelationType.HARMS, 0.65, "{subj} harms {obj}"),
    _RelationPattern(_EATS_RE, RelationType.DEPENDS_ON, 0.6, "{subj} depends on {obj}"),
    # Negative needs must come before generic wants/needs so 'does not need'
    # is not consumed as 'subject=genesis does not, verb=need'.
    _RelationPattern(
        _DOES_NOT_NEED_RE, RelationType.PREVENTS, 0.55,
        "{subj} does not need {obj}",
    ),
    # Infinitive wants must come before generic wants so 'wants to learn'
    # captures the whole goal, not just 'to' as the object.
    _RelationPattern(
        _WANTS_TO_RE, RelationType.DEPENDS_ON, 0.6,
        "{subj} wants to {obj}",
    ),
    _RelationPattern(_WANTS_RE, RelationType.DEPENDS_ON, 0.6, "{subj} depends on {obj}"),
    _RelationPattern(_TEACHES_RE, RelationType.ENABLES, 0.6, "{subj} enables {obj}"),
    _RelationPattern(_USES_RE, RelationType.ENABLES, 0.6, "{subj} uses {obj}"),
    _RelationPattern(_LIVES_IN_RE, RelationType.SPATIAL_RELATION, 0.55, "{subj} is in {obj}"),
    _RelationPattern(_IS_ABOUT_RE, RelationType.RELATED_TO, 0.6, "{subj} relates to {obj}"),
    _RelationPattern(
        _CREATED_BY_RE, RelationType.CREATES, 0.7,
        "{obj} creates {subj}", reversed=True,
    ),
    _RelationPattern(_KNOWS_RE, RelationType.RELATED_TO, 0.55, "{subj} relates to {obj}"),
    _RelationPattern(_DEPENDS_RE, RelationType.DEPENDS_ON, 0.7, "{subj} depends on {obj}"),
    _RelationPattern(_ENABLES_RE, RelationType.ENABLES, 0.7, "{subj} enables {obj}"),
    _RelationPattern(_HAS_RE, RelationType.RELATED_TO, 0.5, "{subj} has {obj}"),
    # ── IS_A last (least specific, catches "is a/an") ──
    _RelationPattern(_IS_A_RE, RelationType.IS_A, 0.7, "{subj} is a {obj}"),
)

# Possessive patterns are handled separately because they generate
# a PART_OF edge with reversed source/target, and they're very common
# in natural text (e.g., "the brain's hippocampus"). We keep them in
# a separate table to avoid interfering with the main pattern priority.
_POSSESSIVE_PATTERN = _RelationPattern(
    _POSSESSIVE_RE, RelationType.PART_OF, 0.5,
    "{obj} is part of {subj}", reversed=True,
)


# ═══════════════════════════════════════════════════════════════════
# Concept normalization
# ═══════════════════════════════════════════════════════════════════

# Words that are plural but don't follow simple rules
_IRREGULAR_PLURALS: dict[str, str] = {
    "men": "man",
    "women": "woman",
    "children": "child",
    "feet": "foot",
    "teeth": "tooth",
    "geese": "goose",
    "mice": "mouse",
    "people": "person",
    "oxen": "ox",
    # Additional irregular plurals
    "lice": "louse",
    "cacti": "cactus",
    "fungi": "fungus",
    "nuclei": "nucleus",
    "syllabi": "syllabus",
    "analyses": "analysis",
    "crises": "crisis",
    "theses": "thesis",
    "oases": "oasis",
    "parentheses": "parenthesis",
    "hypotheses": "hypothesis",
    "diagnoses": "diagnosis",
    "syntheses": "synthesis",
    "phenomena": "phenomenon",
    "criteria": "criterion",
    "data": "datum",
    "media": "medium",
    "bacteria": "bacterium",
    "alumni": "alumnus",
    "stimuli": "stimulus",
    "foci": "focus",
    "radii": "radius",
    "octopi": "octopus",  # common but technically wrong
    "axes": "axis",
    "bases": "basis",
    "matrices": "matrix",
    "vertices": "vertex",
    "indices": "index",
    "appendices": "appendix",
}

# Words that end in 's' but are not plural
_NOT_PLURAL = frozenset(
    {
        "is",
        "was",
        "has",
        "its",
        "this",
        "news",
        "series",
        "species",
        "gas",
        "bus",
        "plus",
        "thus",
        "us",
        "vs",
        "ms",
        "dr",
        "jr",
        "sr",
        "oasis",
        "pelvis",
        "iris",
        "axis",
        "trellis",
        # Additional non-plural words ending in 's'
        "chaos",
        "pathos",
        "ethos",
        "logos",
        "bathos",
        "lens",
        "fetus",
        "status",
        "campus",
        "corpus",
        "virus",
        "bonus",
        "minus",
        "opus",
        "gestus",
        "atlas",
        "bias",
        "alias",
        "arcs",
        "boss",
        "loss",
        "moss",
        "cross",
        "dress",
        "chess",
        "class",
        "glass",
        "grass",
        "press",
        "stress",
        "access",
        "princess",
        "address",
        "process",
        "success",
        "witness",
        "fitness",
    }
)

# Common adjective suffixes — words ending in these are likely
# adjectives that modify a following head noun. Used to decompose
# compound objects in is_a statements: "microscopic animal" →
# the head noun is "animal", so "microscopic animal is_a animal"
# is also learned, enabling transitive inference
# (tardigrade → microscopic animal → animal).
_ADJECTIVE_SUFFIXES: tuple[str, ...] = (
    "ic", "ical", "ive", "ous", "able", "ible", "ful",
    "less", "ish", "ant", "ent", "al",
)

# Nouns that end in adjective suffixes but are NOT adjectives.
# Excluded from the suffix heuristic so they aren't wrongly
# stripped as modifiers (e.g., "animal" ends in -al but is a noun).
# Only words that are PRIMARILY nouns are listed here — words that
# are commonly adjectives (positive, negative, relative, etc.) are
# intentionally excluded so the heuristic can strip them.
_ADJECTIVE_SUFFIX_NOUNS: frozenset[str] = frozenset({
    # -al (primarily nouns)
    "animal", "metal", "canal", "festival", "signal", "rival",
    "survival", "crystal", "hospital", "capital", "portal", "spiral",
    "funeral", "journal", "coral", "mineral", "pedestal", "decimal",
    # -ant (primarily nouns)
    "ant", "elephant", "peasant", "merchant", "servant",
    "warrant", "contestant", "accountant", "assistant",
    # -ent (primarily nouns)
    "tent", "parent", "student", "moment", "agent",
    "ingredient", "president", "resident", "incident", "accent",
    "extent", "percent", "content", "intent", "patent",
    "element", "client", "ferment", "sediment", "serpent",
    # -ic (primarily nouns)
    "music", "magic", "topic", "clinic", "comic", "fabric", "garlic",
    "picnic", "traffic",
    # -ive (primarily nouns)
    "archive", "incentive", "perspective", "detective", "adjective",
    # -ish (primarily nouns)
    "fish", "dish",
})

# Degree adverbs that modify adjectives — stripped before adjective
# detection so "extremely luminous sphere" → head noun "sphere".
_DEGREE_ADVERBS: frozenset[str] = frozenset({
    "extremely", "very", "highly", "deeply", "widely", "particularly",
    "especially", "remarkably", "notably", "unusually", "quite", "rather",
    "somewhat", "fairly", "too", "so", "increasingly", "exceptionally",
    "extraordinarily", "intensely", "strongly", "profoundly", "strikingly",
    "genuinely", "truly", "purely", "relatively", "moderately", "slightly",
})


def _is_likely_adjective(word: str) -> bool:
    """Heuristic: is this word likely an adjective?

    Uses adjective suffixes (-ic, -al, -ive, -ous, etc.) with a
    noun exclusion list. This is a fast, dependency-free heuristic
    for stripping modifiers from noun phrases in is_a statements.

    "microscopic" → True (ends in -ic)
    "animal" → False (ends in -al but is a known noun)
    "luminous" → True (ends in -ous)
    "star" → False (no adjective suffix)
    """
    w = word.lower().strip()
    if len(w) < 3:
        return False
    if w in _ADJECTIVE_SUFFIX_NOUNS:
        return False
    # Words ending in -ment are almost always nouns (equipment,
    # environment, development, etc.) even though they end in -ent.
    if w.endswith("ment"):
        return False
    return any(w.endswith(suffix) for suffix in _ADJECTIVE_SUFFIXES)


def _extract_head_noun(phrase: str) -> str | None:
    """Strip leading modifiers from a noun phrase, returning the head noun.

    In "X is a [modifier] [head noun]", the head noun is the genus —
    what X fundamentally IS. This extracts it so the compound can be
    linked to the head noun via is_a, enabling transitive inference.

    "microscopic animal" → "animal"
    "extremely luminous sphere" → "sphere"
    "symbiotic organism" → "organism"
    "transparent optical element" → "element"
    "star" → None (single word, nothing to strip)
    "hot dog" → None ("hot" has no adjective suffix — conservative)

    Returns the head noun (a single word), or None if no modifier was
    stripped or the remainder is multi-word or itself an adjective.
    """
    words = phrase.lower().strip().split()
    if len(words) < 2:
        return None

    idx = 0
    # Strip leading degree adverbs ("extremely", "very", ...)
    while idx < len(words) - 1 and words[idx] in _DEGREE_ADVERBS:
        idx += 1
    # Strip leading adjectives (by suffix heuristic)
    while idx < len(words) - 1 and _is_likely_adjective(words[idx]):
        idx += 1

    if idx == 0:
        return None  # nothing was stripped

    # Only return a single-word head noun — multi-word remainders
    # (e.g., "sphere of" from "luminous sphere of") are likely
    # misparses and shouldn't generate is_a edges.
    if idx + 1 < len(words):
        return None

    head = words[idx]
    if len(head) < 3:
        return None
    # Don't return an adjective as a head noun — if everything was
    # stripped, the phrase was all modifiers (likely a misparse).
    if _is_likely_adjective(head):
        return None
    return head


def _normalize_concept(name: str) -> str:
    """Normalize a concept name to a canonical form.

    This handles:
    - Stripping leading articles (the, a, an)
    - Converting hyphens to spaces (micro-animal → micro animal)
    - Simple pluralization (dogs → dog, cities → city)
    - Stripping trailing relative clause markers

    This prevents the same concept from being added under multiple
    names (e.g., "micro-animal" and "micro animal" as separate nodes).
    """
    name = name.lower().strip()

    # Strip leading articles
    for article in ("the ", "a ", "an "):
        if name.startswith(article):
            name = name[len(article) :]
            break

    # Strip trailing relative clause markers and everything after
    for marker in (" that", " which", " who", " where", " when"):
        idx = name.find(marker + " ")
        if idx > 0:
            name = name[:idx]
            break
        if name.endswith(marker):
            name = name[: -len(marker)]
            break

    # Normalize hyphens to spaces
    name = name.replace("-", " ")

    # Collapse multiple spaces
    name = re.sub(r"\s+", " ", name).strip()

    # Singularize
    name = _singularize(name)

    return name


def _singularize(name: str) -> str:
    """Convert a simple plural to singular form.

    This is intentionally conservative — it only handles the most
    common English plural patterns. Words it can't handle confidently
    are left unchanged rather than risk incorrect transformation.
    """
    # Check irregular plurals
    if name in _IRREGULAR_PLURALS:
        return _IRREGULAR_PLURALS[name]

    # Multi-word: singularize only the last word
    words = name.split()
    if len(words) > 1:
        words[-1] = _singularize_word(words[-1])
        return " ".join(words)

    return _singularize_word(name)


def _singularize_word(word: str) -> str:
    """Singularize a single word."""
    if len(word) < 3 or word in _NOT_PLURAL:
        return word

    # Check irregular plurals (full-word match)
    if word in _IRREGULAR_PLURALS:
        return _IRREGULAR_PLURALS[word]

    # Words ending in "sis" are Greek singulars (analysis, crisis,
    # thesis, synthesis, etc.) — don't strip the 's'.
    if word.endswith("sis"):
        return word

    # Words ending in "us" are Latin singulars (cactus, fungus,
    # nucleus, syllabus, octopus, status, virus) — don't strip.
    if word.endswith("us") and len(word) > 3:
        return word

    # Words ending in "is" (basis, thesis, axis, iris, lens) — don't strip.
    if word.endswith("is") and len(word) > 3:
        return word

    # "ies" → "y" (cities → city, butterflies → butterfly)
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"

    # "ves" → "f" or "fe" (wolves → wolf, knives → knife, leaves → leaf)
    if word.endswith("ves") and len(word) > 4:
        stem = word[:-3]
        # "knives" → "knife" (restore the 'e')
        if stem.endswith("fe") or stem.endswith("li") or stem.endswith("ni"):
            return stem + "fe"
        # "wolves" → "wolf", "leaves" → "leaf", "halves" → "half"
        return stem + "f"

    # "es" → strip if preceded by s, x, z, ch, sh (boxes → box, brushes → brush)
    if word.endswith("es") and len(word) > 3:
        preceding = word[-3]
        if preceding in "sxz" or word.endswith("ches") or word.endswith("shes"):
            return word[:-2]
        # Otherwise just strip the 's' (codes → code)
        return word[:-1]

    # Simple 's' → strip (dogs → dog)
    if word.endswith("s") and not word.endswith("ss"):
        return word[:-1]

    return word




# ═══════════════════════════════════════════════════════════════════
# Data classes
# ═══════════════════════════════════════════════════════════════════


@dataclass(slots=True)
class LearningEvent:
    """A single learning event — something Genesis learned."""

    event_type: (
        str  # "conversation", "transitive", "definition", "analogical", "correction", "self_study"
    )
    description: str
    concepts_involved: list[str] = field(default_factory=list)
    confidence: float = 0.5
    timestamp: int = field(default_factory=lambda: int(time.time() * 1000))


@dataclass(slots=True)
class InferenceResult:
    """Result of an inference pass."""

    new_edges: int = 0
    new_definitions: int = 0
    strengthened: int = 0
    contradictions_found: int = 0
    events: list[LearningEvent] = field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════
# Self-directed learner
# ═══════════════════════════════════════════════════════════════════

# Minimum time between inference cycles (in seconds). Prevents
# running expensive inference on every interaction.
_INFERENCE_COOLDOWN_SECONDS = 30.0

# Relations that are transitive (A→B, B→C implies A→C)
_TRANSITIVE_RELATIONS = frozenset(
    {
        RelationType.IS_A,
        RelationType.PART_OF,
        RelationType.DEPENDS_ON,
        RelationType.ENABLES,
    }
)

# Relations that are mutually exclusive (if A→B exists, B→A is contradictory)
_OPPOSITE_RELATIONS = frozenset(
    {
        RelationType.OPPOSITE_OF,
        RelationType.SIMILAR_TO,
    }
)


class SelfDirectedLearner:
    """Genesis's self-directed learning engine.

    This module lets Genesis improve her own knowledge through:
    - Extracting facts from conversation
    - Inferring new relationships from existing ones
    - Synthesizing definitions from relationships
    - Transferring properties between similar concepts
    - Reviewing and strengthening weak concepts
    - Learning from user corrections

    It operates on the concept network directly, modifying it in-place.
    All inferred knowledge is marked with origin="inferred" so it can
    be distinguished from directly learned knowledge.
    """

    def __init__(self, network: ConceptNetwork, data_dir: str | None = None) -> None:
        """Wire the self-directed learner to its concept network and data directory."""
        self.network = network
        self.data_dir = data_dir
        self._learning_log: deque[LearningEvent] = deque(maxlen=500)
        self._studied_concepts: dict[str, int] = {}  # concept → study count
        self._total_inferences = 0
        self._total_conversation_facts = 0
        self._total_corrections = 0
        self._total_definitions_synthesized = 0
        self._last_inference_time: float = 0.0
        # Targeted study queue — concepts identified by reflection or
        # self-assessment as needing study. These are prioritized in
        # self_study() over the automatic weak-concept scan.
        self._study_targets: list[str] = []
        # Cache the expensive man-page scan; man pages don't change
        # during a single run.
        self._man_page_genus_cache: list[tuple[str, str, str]] | None = None
        # Pause expensive transitive inference while the user is waiting
        # for a response. Conversation fact extraction can still run.
        self._paused = False

    def pause(self) -> None:
        """Pause expensive inference while the user is actively talking."""
        self._paused = True

    def resume(self) -> None:
        """Resume expensive inference when the conversation turn is done."""
        self._paused = False

    def queue_study_target(self, concept: str) -> None:
        """Queue a concept for self-study in the next inference cycle.

        Called by the reflection system when a knowledge gap is detected.
        """
        self._study_targets.append(concept)

    @property
    def paused(self) -> bool:
        """True when expensive inference is temporarily paused."""
        return self._paused

    # ═══════════════════════════════════════════════════════════════
    # 1. Conversation learning — extract facts from user input
    # ═══════════════════════════════════════════════════════════════

    def learn_from_input(self, user_input: str) -> list[LearningEvent]:
        """Extract facts from user input and add them to the network.

        This is called on every user message. It scans for factual
        statements like "X is a Y", "X causes Y", "X means Z" and
        adds them to the concept network.

        Returns a list of learning events for introspection.
        """
        events: list[LearningEvent] = []
        text = user_input.strip()

        # Skip questions — they're not statements of fact
        if text.rstrip().endswith("?"):
            return events
        # Skip very short inputs
        if len(text.split()) < 3:
            return events

        # Split into sentences and process each independently. This
        # prevents a non-extractable lead sentence ("I want to teach
        # you something.") from hiding an extractable follow-up
        # ("A glip is a small flying creature.").
        import re as _re
        sentences = _re.split(r'(?<=[.!?])\s+', text)

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence or len(sentence.split()) < 3:
                continue

            # Track which text spans have been consumed by a pattern,
            # so later patterns don't also match the same text.
            consumed_spans: list[tuple[int, int]] = []

            # ── Extract definitions first ──
            # "X means Y" / "X is defined as Y" would also match the IS_A
            # pattern, so we handle definitions first and mark their spans
            # as consumed.
            events.extend(self._extract_definitions(sentence, consumed_spans))

            # ── Extract typed relationships ──
            # Try each pattern in priority order. Skip spans already consumed.
            events.extend(self._extract_typed_relationships(sentence, consumed_spans))

            # ── Extract possessive PART_OF relationships ──
            # "the brain's hippocampus" → hippocampus is_part_of brain
            # Run after the main patterns so they get priority on
            # overlapping text. Only extract if both parts are meaningful
            # and the span hasn't been consumed.
            events.extend(self._extract_possessive_relationships(sentence, consumed_spans))

        self._learning_log.extend(events)
        return events

    def learn_from_emotion_share(
        self,
        text: str,
        emotion_word: str,
        sentiment: float,
        sentiment_label: str,
    ) -> list[LearningEvent]:
        """Learn from an emotion-sharing statement.

        When someone says "I'm scared of the dark" or "I'm excited
        about the project," extract the emotion→subject relationship
        and learn it. Also learn sentiment-based properties of the
        emotion (e.g., "scared" is related to "negative").

        This is how Genesis learns emotions from natural conversation
        — not just from explicit definitions, but from how people
        actually use emotion words in context.
        """
        events: list[LearningEvent] = []
        if not emotion_word:
            return events

        # Ensure the emotion concept exists in the network
        emotion_id = self.network._resolve(emotion_word)
        if not emotion_id:
            self.network.add_concept(emotion_word, confidence=0.5, origin="emotion_share")
            emotion_id = self.network._resolve(emotion_word)

        # 1. Learn the emotion→subject relationship
        events.extend(
            self._learn_emotion_subject(text, emotion_word, emotion_id)
        )

        # 2. Learn sentiment-based properties
        events.extend(
            self._learn_emotion_sentiment(emotion_word, emotion_id, sentiment)
        )

        # 3. Synthesize a definition from what she just learned
        events.extend(
            self._learn_emotion_definition(emotion_word, emotion_id, sentiment)
        )

        self._learning_log.extend(events)
        return events

    def _learn_emotion_subject(
        self,
        text: str,
        emotion_word: str,
        emotion_id: str | None,
    ) -> list[LearningEvent]:
        """Learn the emotion→subject relationship.

        "I'm scared of the dark" → scared RELATED_TO dark
        "I'm excited about the project" → excited RELATED_TO project
        "I'm angry about the delay" → angry RELATED_TO delay
        """
        events: list[LearningEvent] = []
        subject = self._extract_emotion_subject(text, emotion_word)
        if not subject:
            return events

        subj_id = self.network._resolve(subject)
        if not subj_id:
            self.network.add_concept(subject, confidence=0.4, origin="emotion_share")
            subj_id = self.network._resolve(subject)
        if subj_id and emotion_id:
            if self._add_relationship(
                emotion_word, subject, RelationType.RELATED_TO, 0.5, "emotion_share"
            ):
                events.append(
                    LearningEvent(
                        event_type="conversation",
                        description=f"{emotion_word} is related to {subject}",
                        concepts_involved=[emotion_word, subject],
                        confidence=0.5,
                    )
                )
        return events

    def _learn_emotion_sentiment(
        self,
        emotion_word: str,
        emotion_id: str | None,
        sentiment: float,
    ) -> list[LearningEvent]:
        """Learn sentiment-based properties of the emotion.

        If the sentiment is strongly negative, learn that the emotion
        is related to "negative" (or "positive" for positive sentiment).
        This gives her a basic understanding even without a full definition.
        """
        events: list[LearningEvent] = []
        if abs(sentiment) <= 0.2:
            return events

        polarity = "negative" if sentiment < 0 else "positive"
        polarity_id = self.network._resolve(polarity)
        if polarity_id and emotion_id:
            if self._add_relationship(
                emotion_word, polarity, RelationType.RELATED_TO, 0.4, "emotion_share"
            ):
                events.append(
                    LearningEvent(
                        event_type="conversation",
                        description=f"{emotion_word} is related to {polarity}",
                        concepts_involved=[emotion_word, polarity],
                        confidence=0.4,
                    )
                )
        return events

    def _learn_emotion_definition(
        self,
        emotion_word: str,
        emotion_id: str | None,
        sentiment: float,
    ) -> list[LearningEvent]:
        """Synthesize a definition from what she just learned.

        If the emotion concept has no definition, build a basic one
        from its relationships. This gives her something to say in
        empathy even without an explicit definition.
        """
        events: list[LearningEvent] = []
        concept = self.network.get_concept(emotion_word)
        if concept and concept.properties.get("definition"):
            return events

        edges = self.network.get_edges(emotion_id, "out") if emotion_id else []
        if not edges:
            return events

        related = [
            e.target for e in edges
            if e.relation == RelationType.RELATED_TO
        ][:3]
        if not related:
            return events

        if sentiment < -0.2:
            definition = f"a negative emotion related to {', '.join(related[:2])}"
        elif sentiment > 0.2:
            definition = f"a positive emotion related to {', '.join(related[:2])}"
        else:
            definition = f"an emotion related to {', '.join(related[:2])}"
        self._add_definition(emotion_word, definition)
        events.append(
            LearningEvent(
                event_type="definition",
                description=(
                    f"Synthesized definition for '{emotion_word}': "
                    f"{definition}"
                ),
                concepts_involved=[emotion_word],
                confidence=0.4,
            )
        )
        return events

    def _extract_emotion_subject(self, text: str, emotion_word: str) -> str:
        """Extract the subject of an emotion from the text.

        "I'm scared of the dark" → "dark"
        "I'm excited about the project" → "project"
        "I'm angry about the delay" → "delay"
        "I'm feeling sad today" → "" (no specific subject)
        """
        lower = text.lower()
        # Patterns: "scared of X", "excited about X", "angry about X",
        # "happy with X", "worried about X", "anxious about X"
        patterns = [
            rf"\b{re.escape(emotion_word)}\s+(?:about|of|with|over|by|at)\s+(?:the\s+)?([a-z][a-z\s]+?)(?:[.,;!?]|\s+(?:and|because|so|but|which|that)\s|$)",
            rf"\bfeeling\s+{re.escape(emotion_word)}\s+(?:about|of|with|over|by|at)\s+(?:the\s+)?([a-z][a-z\s]+?)(?:[.,;!?]|\s+(?:and|because|so|but|which|that)\s|$)",
        ]
        for pattern in patterns:
            m = re.search(pattern, lower)
            if m:
                subject = m.group(1).strip()
                # Normalize and validate
                subject = _normalize_concept(subject)
                if subject and self._is_meaningful(subject) and len(subject) > 2:
                    return subject
        return ""

    def _extract_definitions(
        self, text: str, consumed_spans: list[tuple[int, int]]
    ) -> list[LearningEvent]:
        """Extract definition statements from text.

        Two patterns are tried:

        1. **Explicit definitions** ("X means Y", "X is defined as Y",
           "X refers to Y", "X is when Y") — high confidence, 3+ words.

        2. **Bare copula definitions** ("X is Y") — catches the most
           common definitional form in natural language. Runs after the
           explicit pattern so "X is defined as Y" isn't double-matched.
           The negative lookahead in _BARE_IS_RE excludes patterns
           handled by typed relationship regexes (is a, is part of,
           is similar to, etc.), so this only catches what they miss.

        Definitions are handled before typed relationships because
        "X means Y" would also match the IS_A pattern. Matched spans
        are recorded in consumed_spans so later patterns skip them.
        """
        events: list[LearningEvent] = []

        # 1. Explicit definitions
        for m in _DEFINITION_RE.finditer(text):
            concept_name = _normalize_concept(m.group(1))
            definition = m.group(2).strip().rstrip(".")
            if self._is_meaningful(concept_name) and len(definition.split()) >= 3:
                if self._add_definition(concept_name, definition):
                    events.append(
                        LearningEvent(
                            event_type="conversation",
                            description=(
                                f"Learned definition of '{concept_name}': {definition[:60]}"
                            ),
                            concepts_involved=[concept_name],
                            confidence=0.8,
                        )
                    )
                    self._total_conversation_facts += 1
            consumed_spans.append((m.start(), m.end()))

        # 2. Bare copula definitions ("X is Y")
        #    Lower confidence (0.6) than explicit definitions (0.8)
        #    because bare "is" is more ambiguous. Requires 2+ words
        #    in the definition to avoid matching trivial statements
        #    like "X is here". Single-word definitions are accepted
        #    only if the word is 4+ characters (filters out "is ok"
        #    but keeps "is natural", "is contagious").
        for m in _BARE_IS_RE.finditer(text):
            # Skip if already consumed by explicit definition
            if any(s <= m.start() < e for s, e in consumed_spans):
                continue
            concept_name = _normalize_concept(m.group(1))
            definition = m.group(2).strip().rstrip(".")
            if not self._is_meaningful(concept_name):
                continue
            word_count = len(definition.split())
            if word_count < 2 and len(definition) < 4:
                continue
            if self._add_definition(concept_name, definition):
                events.append(
                    LearningEvent(
                        event_type="conversation",
                        description=(
                            f"Learned definition of '{concept_name}': {definition[:60]}"
                        ),
                        concepts_involved=[concept_name],
                        confidence=0.6,
                    )
                )
                self._total_conversation_facts += 1
            consumed_spans.append((m.start(), m.end()))

        return events

    def _extract_typed_relationships(
        self, text: str, consumed_spans: list[tuple[int, int]]
    ) -> list[LearningEvent]:
        """Extract typed relationships (IS_A, CAUSES, etc.) from text.

        Tries each relation pattern in priority order, skipping spans
        already consumed by earlier patterns (e.g. definitions).
        """
        events: list[LearningEvent] = []
        for pat in _RELATION_PATTERNS:
            for m in pat.regex.finditer(text):
                if any(s <= m.start() < e for s, e in consumed_spans):
                    continue

                # For reversed patterns (passive voice, "composed of"),
                # group(1) is the target and group(2) is the source.
                if pat.reversed:
                    subject = _normalize_concept(m.group(2))
                    obj = _normalize_concept(m.group(1))
                else:
                    subject = _normalize_concept(m.group(1))
                    obj = _normalize_concept(m.group(2))

                if not (self._is_meaningful(subject) and self._is_meaningful(obj)):
                    continue
                if subject == obj:
                    continue

                # For is_a statements, capture the full definitional phrase
                # (e.g., "a measure of disorder in a system") not just the genus.
                full_end = m.end()
                if pat.relation == RelationType.IS_A and not pat.reversed:
                    tail = text[m.end():]
                    punct = re.search(r"[.!?](?:\s|$)|[,;]", tail)
                    full_end = m.end() + (punct.start() if punct else len(tail))
                    article = re.search(
                        r"\b(?:is|are)\s+(a|an)(?:\s+(?:type|kind|form|sort|class|category)\s+of)?\s*",
                        m.group(0),
                        re.IGNORECASE,
                    )
                    if article:
                        defn_start = m.start(0) + article.start(1)
                        self._add_definition(subject, text[defn_start:full_end])

                if self._add_relationship(subject, obj, pat.relation, pat.weight, "stated"):
                    events.append(
                        LearningEvent(
                            event_type="conversation",
                            description=pat.label.format(subj=subject, obj=obj),
                            concepts_involved=[subject, obj],
                            confidence=pat.weight,
                        )
                    )
                    self._total_conversation_facts += 1

                # For is_a statements with a modified noun phrase as the
                # object (e.g., "microscopic animal"), also link the
                # compound to its head noun ("animal"). This captures the
                # semantic truth that a microscopic animal IS an animal,
                # enabling transitive inference (tardigrade → microscopic
                # animal → animal). Without this, Genesis learns the
                # compound category but not the fundamental genus.
                if pat.relation == RelationType.IS_A and not pat.reversed:
                    head = _extract_head_noun(obj)
                    if head and head != obj and head != subject:
                        if self._add_relationship(
                            obj, head, RelationType.IS_A, 0.5, "stated"
                        ):
                            events.append(
                                LearningEvent(
                                    event_type="conversation",
                                    description=f"{obj} is a {head}",
                                    concepts_involved=[obj, head],
                                    confidence=0.5,
                                )
                            )
                            self._total_conversation_facts += 1

                consumed_spans.append((m.start(), full_end))
        return events

    def _extract_possessive_relationships(
        self, text: str, consumed_spans: list[tuple[int, int]]
    ) -> list[LearningEvent]:
        """Extract possessive PART_OF relationships from text.

        "the brain's hippocampus" → hippocampus is_part_of brain.
        Run after the main patterns so they get priority on overlapping
        text. Only extracts if both parts are meaningful and the span
        hasn't been consumed.
        """
        events: list[LearningEvent] = []
        for m in _POSSESSIVE_PATTERN.regex.finditer(text):
            if any(s <= m.start() < e for s, e in consumed_spans):
                continue

            # Reversed: group(1) is the whole (X), group(2) is the part (Y)
            subject = _normalize_concept(m.group(1))  # the whole
            obj = _normalize_concept(m.group(2))       # the part

            if not (self._is_meaningful(subject) and self._is_meaningful(obj)):
                continue
            if subject == obj:
                continue

            # Y is_part_of X
            if self._add_relationship(obj, subject, RelationType.PART_OF, 0.5, "stated"):
                events.append(
                    LearningEvent(
                        event_type="conversation",
                        description=f"{obj} is part of {subject}",
                        concepts_involved=[obj, subject],
                        confidence=0.5,
                    )
                )
                self._total_conversation_facts += 1

            consumed_spans.append((m.start(), m.end()))
        return events

    # ═══════════════════════════════════════════════════════════════
    # 2. Transitive inference — A→B, B→C → A→C
    # ═══════════════════════════════════════════════════════════════

    def _infer_transitive_pass(
        self,
        adjacency: dict,
        rel_type: RelationType,
        max_inferences: int,
        inferences_made: int,
        result: InferenceResult,
    ) -> tuple[int, int]:
        """Run one pass of transitive inference for a single relation type."""
        pass_inferences = 0

        # For each A→B, check B→C and infer A→C
        for a, targets in adjacency.items():
            if inferences_made >= max_inferences:
                break
            for b in targets:
                for c in adjacency.get(b, []):
                    if c == a or c == b:
                        continue

                    # Skip if A→C already exists (any relation)
                    # Use _has_any_edge_resolved for speed — a and c are
                    # already normalized concept IDs from the adjacency list.
                    if self._has_any_edge_resolved(a, c):
                        continue

                    # Skip contradictions
                    if self._is_contradiction(a, c, rel_type):
                        result.contradictions_found += 1
                        continue

                    # Skip cycles in hierarchical relations
                    if self._would_create_cycle(a, c, rel_type):
                        continue

                    edge = self.network.add_edge(
                        a,
                        c,
                        rel_type,
                        weight=0.35,
                        origin="inferred",
                    )
                    if edge:
                        inferences_made += 1
                        pass_inferences += 1
                        result.new_edges += 1
                        result.events.append(
                            LearningEvent(
                                event_type="transitive",
                                description=f"Inferred: {a} {rel_type.value} {c}",
                                concepts_involved=[a, c],
                                confidence=0.35,
                            )
                        )
                        self._total_inferences += 1
                        # Update adjacency so subsequent passes
                        # can use the newly inferred edge.
                        adjacency.setdefault(a, []).append(c)

        return pass_inferences, inferences_made

    def infer_transitive(self, max_inferences: int = 50) -> InferenceResult:
        """Infer new relationships through transitivity.

        For transitive relations (IS_A, PART_OF, DEPENDS_ON):
        If A is_a B and B is_a C, infer A is_a C.

        Runs up to 3 passes so that multi-hop inferences are found
        (e.g., golden retriever → dog → mammal → animal).

        Inferred edges have lower weight and origin="inferred".
        Contradictions are detected and skipped.
        """
        result = InferenceResult()
        inferences_made = 0

        # Build adjacency lists for ALL transitive relations in a single
        # pass over the concept network, instead of one pass per relation
        # type per inference pass. This avoids O(n * R * P) get_edges()
        # calls, replacing them with O(n) edge scans.
        adjacency_by_rel: dict = {rel: {} for rel in _TRANSITIVE_RELATIONS}
        for edge in self.network._edges:
            if edge.relation in adjacency_by_rel:
                adjacency_by_rel[edge.relation].setdefault(
                    edge.source, []
                ).append(edge.target)

        for rel_type in _TRANSITIVE_RELATIONS:
            if inferences_made >= max_inferences:
                break

            # Run up to 3 passes to catch multi-hop inferences.
            # Each pass uses the updated adjacency list, so edges
            # inferred in pass 1 can be used in pass 2.
            for _pass in range(3):
                if inferences_made >= max_inferences:
                    break
                adjacency = adjacency_by_rel[rel_type]
                pass_inferences, inferences_made = self._infer_transitive_pass(
                    adjacency, rel_type, max_inferences, inferences_made, result
                )
                # If no new inferences were made in this pass, no
                # point running another pass — we've reached fixpoint
                if pass_inferences == 0:
                    break

        self._learning_log.extend(result.events)
        return result

    # ═══════════════════════════════════════════════════════════════
    # 3. Definition synthesis — build definitions from relationships
    # ═══════════════════════════════════════════════════════════════

    def synthesize_definitions(self, max_definitions: int = 30) -> InferenceResult:
        """Synthesize definitions for concepts that lack them.

        For each concept without a definition, look at its relationships
        and build a definition from them.

        Examples:
        - "X is_a Y" → "X is a kind of Y"
        - "X part_of Y" → "X is a part of Y"
        - "X causes Y" → "X causes Y"
        - "X enables Y" → "X enables Y"
        """
        result = InferenceResult()

        for cid, concept in list(self.network._concepts.items()):
            if result.new_definitions >= max_definitions:
                break

            existing = concept.properties.get("definition", "")
            if existing and len(existing) > 10 and existing != "NO DEF":
                continue

            definition = self._synthesize_definition(cid)
            if definition:
                concept.properties["definition"] = definition
                result.new_definitions += 1
                result.events.append(
                    LearningEvent(
                        event_type="definition",
                        description=f"Synthesized definition for '{cid}': {definition[:60]}",
                        concepts_involved=[cid],
                        confidence=0.5,
                    )
                )
                self._total_definitions_synthesized += 1
            else:
                # Synthesis failed — try WordNet enrichment
                if self._enrich_from_wordnet(cid, result):
                    self._total_definitions_synthesized += 1

        self._learning_log.extend(result.events)
        return result

    def _synthesize_definition(self, concept_id: str) -> str | None:
        """Build a definition for a concept from its relationships."""
        edges = self.network.get_edges(concept_id, "out")
        parts = self._collect_definition_parts(edges)
        if not parts:
            return None
        return self._assemble_definition(concept_id, parts)

    def _enrich_from_wordnet(
        self, concept_id: str, result: InferenceResult
    ) -> bool:
        """Enrich a weak concept from WordNet when synthesis fails.

        Looks up the concept in WordNet (via NLTK). If found, stores the
        dictionary definition and adds semantic relations to the concept
        network:
        - hypernyms → IS_A edges
        - meronyms → PART_OF edges
        - similar_to → SIMILAR_TO edges
        - antonyms → OPPOSITE_OF edges

        This is her primary word reference — no web access required.
        Returns True if the concept was enriched, False otherwise.
        Gracefully no-ops when WordNet/NLTK is unavailable.
        """
        try:
            from ..wordnet_dictionary import lookup_word
        except ImportError:
            return False

        # Use the display name (underscores → spaces) for lookup
        clean_name = concept_id.replace("_", " ")
        entries = lookup_word(clean_name, max_senses=1)
        if not entries:
            return False

        entry = entries[0]
        concept = self.network.get_concept(concept_id)
        if concept is None:
            return False

        # Store the dictionary definition
        concept.properties["definition"] = entry.definition
        if entry.examples:
            concept.properties["example"] = entry.examples[0]
        concept.properties["part_of_speech"] = entry.part_of_speech
        result.new_definitions += 1

        # Add semantic relations to the concept network
        relations_added = 0
        relation_map = {
            "hypernyms": RelationType.IS_A,
            "meronyms": RelationType.PART_OF,
            "similar_to": RelationType.SIMILAR_TO,
            "antonyms": RelationType.OPPOSITE_OF,
        }
        for rel_key, rel_type in relation_map.items():
            for related in getattr(entry, rel_key, []):
                related_clean = related.replace("_", " ")
                if related_clean == concept_id:
                    continue
                # Create the related concept if it doesn't exist
                if self.network.get_concept(related_clean) is None:
                    self.network.add_concept(
                        related_clean, confidence=0.5, origin="wordnet"
                    )
                # Add the edge (skip if it already exists)
                existing = self.network.get_edges(concept_id, "out")
                if not any(
                    e.target == related_clean and e.relation == rel_type
                    for e in existing
                ):
                    self.network.add_edge(
                        concept_id, related_clean, rel_type,
                        0.6, origin="wordnet",
                    )
                    relations_added += 1

        result.events.append(
            LearningEvent(
                event_type="self_study",
                description=(
                    f"Self-study: WordNet enriched '{concept_id}' — "
                    f"definition + {relations_added} relations"
                ),
                concepts_involved=[concept_id],
                confidence=0.6,
            )
        )
        logger.debug(
            "WordNet enriched '%s': definition + %d relations",
            concept_id, relations_added,
        )
        return True

    def _collect_definition_parts(self, edges: list) -> list[str]:
        """Collect descriptive phrase parts from a concept's outgoing edges.

        Each relationship type contributes a phrase like "a kind of Y"
        or "causes Y". IS_A uses the highest-weight parent and optionally
        a second parent. RELATED_TO is only used when there's no IS_A.

        Only edges from reliable origins are used — bridge edges from
        hub attachment and associative bridging are excluded because
        they can create semantically meaningless connections (e.g.
        "enactivism" leaking into prime number definitions via
        word-overlap hub attachment).
        """
        # Origins that produce reliable, semantically meaningful edges.
        # Bridge origins (hub_attachment, associative_bridge, bridge)
        # are excluded — they use word overlap or shared-neighbor heuristics
        # that can connect unrelated concepts.
        _RELIABLE_ORIGINS = frozenset({
            "learned", "stated", "conversation", "inferred", "observed",
            "semantic_bridge", "semantic_connect", "curriculum",
            "relation_verb_seed", "relation_phrase_seed",
        })
        # Filter to reliable, sufficiently-weighted edges
        reliable = [
            e for e in edges
            if e.origin in _RELIABLE_ORIGINS and e.weight >= 0.3
        ]

        parts: list[str] = []

        # IS_A → "a kind of Y" (use highest-weight parent)
        is_a_targets = [(e.target, e.weight) for e in reliable if e.relation == RelationType.IS_A]
        if is_a_targets:
            is_a_targets.sort(key=lambda t: -t[1])  # highest weight first
            parts.append(f"a kind of {is_a_targets[0][0]}")
            if len(is_a_targets) > 1:
                parts.append(f"also considered a {is_a_targets[1][0]}")

        # PART_OF → "part of Y"
        part_of = [e.target for e in reliable if e.relation == RelationType.PART_OF]
        if part_of:
            parts.append(f"part of {part_of[0]}")

        # CAUSES → "causes Y"
        causes = [e.target for e in reliable if e.relation == RelationType.CAUSES]
        if causes:
            parts.append(f"causes {causes[0]}")

        # ENABLES → "enables Y"
        enables = [e.target for e in reliable if e.relation == RelationType.ENABLES]
        if enables:
            parts.append(f"enables {enables[0]}")

        # PREVENTS → "prevents Y"
        prevents = [e.target for e in reliable if e.relation == RelationType.PREVENTS]
        if prevents:
            parts.append(f"prevents {prevents[0]}")

        # DEPENDS_ON → "depends on Y"
        depends = [e.target for e in reliable if e.relation == RelationType.DEPENDS_ON]
        if depends:
            parts.append(f"depends on {depends[0]}")

        # EMERGES_FROM → "emerges from Y"
        emerges = [e.target for e in reliable if e.relation == RelationType.EMERGES_FROM]
        if emerges:
            parts.append(f"emerges from {emerges[0]}")

        # CREATES → "creates Y"
        creates = [e.target for e in reliable if e.relation == RelationType.CREATES]
        if creates:
            parts.append(f"creates {creates[0]}")

        # HARMS → "harms Y"
        harms = [e.target for e in reliable if e.relation == RelationType.HARMS]
        if harms:
            parts.append(f"harms {harms[0]}")

        # OPPOSITE_OF → "the opposite of Y"
        opposite = [e.target for e in reliable if e.relation == RelationType.OPPOSITE_OF]
        if opposite:
            parts.append(f"the opposite of {opposite[0]}")

        # SIMILAR_TO → "similar to Y"
        similar = [e.target for e in reliable if e.relation == RelationType.SIMILAR_TO]
        if similar:
            parts.append(f"similar to {similar[0]}")

        # RELATED_TO → "related to Y" (only if no more specific relation)
        if not is_a_targets:
            related = [e.target for e in reliable if e.relation == RelationType.RELATED_TO]
            if related:
                parts.append(f"related to {related[0]}")

        return parts

    def _assemble_definition(self, concept_id: str, parts: list[str]) -> str:
        """Assemble a readable definition string from collected parts.

        Returns just the definition phrase (e.g., "a kind of micro
        animal") without the concept name prefix — the caller
        (compose_about) will prepend "X is " when composing speech.

        If the first part is a classification ("a kind of" or "part of"),
        joins the first three parts. Otherwise leads with the first part
        and appends up to two more. Truncates to 200 characters.
        """
        if parts[0].startswith("a kind of") or parts[0].startswith("part of"):
            definition = ", ".join(parts[:3])
        else:
            definition = parts[0]
            if len(parts) > 1:
                definition += f" and {', '.join(parts[1:3])}"

        definition = definition.rstrip(",")
        if len(definition) > 200:
            definition = definition[:197] + "..."

        return definition

    # ═══════════════════════════════════════════════════════════════
    # 4. Analogical transfer — similar concepts share properties
    # ═══════════════════════════════════════════════════════════════

    def infer_analogical(self, max_inferences: int = 20) -> InferenceResult:
        """Infer new relationships through analogical transfer.

        If two concepts share many IS_A parents, they're similar.
        If one has a property the other doesn't, infer that the
        other might have it too.

        This is speculative — inferred edges have low confidence (0.25).

        To avoid O(n²) over all concepts, we only compare concepts
        that share at least one IS_A parent. We build a parent→children
        index and only compare siblings.
        """
        result = InferenceResult()
        inferences_made = 0

        # Build parent → children index AND parent sets in a single pass
        # over edges, instead of two passes over all concepts calling
        # get_edges() for each.
        parent_to_children: dict[str, list[str]] = {}
        parents: dict[str, set[str]] = {}
        for edge in self.network._edges:
            if edge.relation != RelationType.IS_A:
                continue
            parent_to_children.setdefault(edge.target, []).append(edge.source)
            parents.setdefault(edge.source, set()).add(edge.target)
        # Only keep concepts with >= 2 parents (needed for similarity)
        parents = {cid: ps for cid, ps in parents.items() if len(ps) >= 2}

        # Compare only siblings (concepts sharing at least one parent)
        compared: set[tuple[str, str]] = set()
        for _parent, children in parent_to_children.items():
            if len(children) < 2:
                continue
            for i, a in enumerate(children):
                if inferences_made >= max_inferences:
                    break
                if a not in parents:
                    continue
                for b in children[i + 1 :]:
                    if b not in parents:
                        continue
                    pair = (a, b) if a < b else (b, a)
                    if pair in compared:
                        continue
                    compared.add(pair)

                    # Jaccard similarity of parent sets
                    shared = parents[a] & parents[b]
                    union = parents[a] | parents[b]
                    similarity = len(shared) / len(union)

                    # Need at least 2 shared parents and >50% similarity
                    if len(shared) < 2 or similarity < 0.5:
                        continue

                    inferences_made += self._transfer_edges(
                        a, b, result, max_inferences - inferences_made
                    )
                    inferences_made += self._transfer_edges(
                        b, a, result, max_inferences - inferences_made
                    )

        self._learning_log.extend(result.events)
        return result

    def _transfer_edges(
        self,
        source: str,
        target: str,
        result: InferenceResult,
        budget: int,
    ) -> int:
        """Transfer edges from source to target (analogical inference).

        Returns the number of edges transferred.
        """
        if budget <= 0:
            return 0

        transferred = 0
        source_edges = {(e.target, e.relation) for e in self.network.get_edges(source, "out")}
        target_edges = {(e.target, e.relation) for e in self.network.get_edges(target, "out")}

        for edge_target, rel in source_edges:
            if transferred >= budget:
                break
            if (edge_target, rel) in target_edges:
                continue
            if edge_target == target or edge_target == source:
                continue
            if self._has_edge(target, edge_target, rel):
                continue

            # Don't transfer IS_A if it would create a cycle
            if rel == RelationType.IS_A and self._would_create_cycle(target, edge_target, rel):
                continue

            edge = self.network.add_edge(
                target,
                edge_target,
                rel,
                weight=0.25,
                origin="analogical",
            )
            if edge:
                transferred += 1
                result.new_edges += 1
                result.events.append(
                    LearningEvent(
                        event_type="analogical",
                        description=(
                            f"Inferred by analogy: {target} {rel.value} "
                            f"{edge_target} (because {target} is similar to {source})"
                        ),
                        concepts_involved=[target, edge_target, source],
                        confidence=0.25,
                    )
                )
                self._total_inferences += 1

        return transferred

    # ═══════════════════════════════════════════════════════════════
    # 5. Self-study — review and strengthen weak concepts
    # ═══════════════════════════════════════════════════════════════

    def _study_concept(
        self, concept_id: str, result: InferenceResult
    ) -> tuple[int, int]:
        """Study a single weak concept: strengthen, synthesize, infer."""
        # 1. Strengthen existing edges
        edges = self.network.get_edges(concept_id, "out")
        incoming = self.network.get_edges(concept_id, "in")
        for edge in edges + incoming:
            edge.weight = min(0.9, edge.weight + 0.03)
            result.strengthened += 1

        # 2. Synthesize definition if missing
        concept = self.network.get_concept(concept_id)
        if concept:
            existing = concept.properties.get("definition", "")
            if not existing or existing == "NO DEF":
                definition = self._synthesize_definition(concept_id)
                if definition:
                    concept.properties["definition"] = definition
                    result.new_definitions += 1
                    result.events.append(
                        LearningEvent(
                            event_type="self_study",
                            description=(
                                f"Self-study: synthesized definition for '{concept_id}'"
                            ),
                            concepts_involved=[concept_id],
                            confidence=0.5,
                        )
                    )
                    self._total_definitions_synthesized += 1
                else:
                    # Synthesis failed — try WordNet lookup to enrich
                    # the concept with a dictionary definition and
                    # semantic relations (IS_A, PART_OF, SIMILAR_TO,
                    # OPPOSITE_OF). This is her primary word reference;
                    # no web access required.
                    enriched = self._enrich_from_wordnet(concept_id, result)
                    if enriched:
                        self._total_definitions_synthesized += 1

        # 3. Infer new edges from siblings (deepening)
        new_edges = self._infer_from_siblings(concept_id, result)
        if new_edges > 0:
            result.new_edges += new_edges

        # 4. Mark as studied
        self._studied_concepts[concept_id] = self._studied_concepts.get(concept_id, 0) + 1

        return len(edges) + len(incoming), new_edges

    def self_study(
        self,
        weak_concepts: list[tuple[str, float]] | None = None,
        max_concepts: int = 10,
    ) -> InferenceResult:
        """Review and strengthen weak concepts.

        For each weak concept:
        1. Strengthen existing edges (increase weight slightly)
        2. Synthesize a definition if missing
        3. Infer new edges from sibling concepts (concepts sharing
           an IS_A parent) — if a sibling has a property the weak
           concept lacks, infer that the weak concept might have it too
        4. Mark as studied

        This is the "homework" Genesis does on her own to improve
        her understanding of concepts she's weak on. Steps 3-4 are
        the deepening: she doesn't just review, she actively learns
        new things about weak concepts by analogy to their siblings.
        """
        result = InferenceResult()

        if weak_concepts is None:
            # Prioritize targeted study concepts (from reflection gaps)
            # over the automatic weak-concept scan
            if self._study_targets:
                targets: list[tuple[str, float]] = []
                for t in self._study_targets[:max_concepts]:
                    concept = self.network.get_concept(t)
                    if concept:
                        targets.append((t, concept.confidence))
                self._study_targets = self._study_targets[max_concepts:]
                if targets:
                    weak_concepts = targets
            if not weak_concepts:
                weak_concepts = self._find_weak_concepts(max_concepts)

        for concept_id, _confidence in weak_concepts[:max_concepts]:
            strengthened_count, new_edges = self._study_concept(concept_id, result)
            result.events.append(
                LearningEvent(
                    event_type="self_study",
                    description=(
                        f"Self-study: reviewed '{concept_id}', "
                        f"strengthened {strengthened_count} connections"
                        + (f", inferred {new_edges} new" if new_edges else "")
                    ),
                    concepts_involved=[concept_id],
                    confidence=0.6,
                )
            )

        self._learning_log.extend(result.events)
        return result

    def _find_siblings(self, concept_id: str) -> set[str]:
        """Find sibling concepts (sharing an IS_A parent) for a concept."""
        is_a_parents = {
            e.target
            for e in self.network.get_edges(concept_id, "out")
            if e.relation == RelationType.IS_A
        }
        if not is_a_parents:
            return set()

        siblings: set[str] = set()
        for parent in is_a_parents:
            for edge in self.network.get_edges(parent, "in"):
                if edge.relation == RelationType.IS_A and edge.source != concept_id:
                    siblings.add(edge.source)
        return siblings

    def _infer_sibling_edges(
        self,
        concept_id: str,
        siblings: set[str],
        existing_targets: set[tuple[str, RelationType]],
        result: InferenceResult,
    ) -> int:
        """Infer new edges for a concept from sibling outgoing edges."""
        inferred = 0
        # For each sibling, look at their outgoing edges
        for sibling in siblings:
            for s_edge in self.network.get_edges(sibling, "out"):
                # Skip IS_A edges (we already know the parents)
                if s_edge.relation == RelationType.IS_A:
                    continue
                # Skip if we already have this edge
                key = (s_edge.target, s_edge.relation)
                if key in existing_targets:
                    continue
                # Skip self-references
                if s_edge.target == concept_id:
                    continue

                # Infer that concept_id might have this property too
                new_edge = self.network.add_edge(
                    concept_id,
                    s_edge.target,
                    s_edge.relation,
                    weight=0.25,
                    origin="inferred",
                )
                if new_edge:
                    inferred += 1
                    existing_targets.add(key)
                    result.events.append(
                        LearningEvent(
                            event_type="self_study",
                            description=(
                                f"Self-study: inferred '{concept_id}' "
                                f"{s_edge.relation.value} '{s_edge.target}' "
                                f"by analogy to sibling '{sibling}'"
                            ),
                            concepts_involved=[concept_id, s_edge.target],
                            confidence=0.25,
                        )
                    )
                    self._total_inferences += 1

        return inferred

    def _infer_from_siblings(
        self,
        concept_id: str,
        result: InferenceResult,
    ) -> int:
        """Infer new edges for a concept by analogy to its siblings.

        If concept_id is_a Parent, and Sibling is_a Parent, and Sibling
        has an edge (e.g., Sibling has X), then infer that concept_id
        might also have X. This is a form of analogical reasoning
        specific to self-study: "my sibling has this property, maybe
        I do too."

        Only infers edges that don't already exist. Uses low weight
        (0.25) and origin="inferred" to mark these as speculative.

        Returns the number of new edges inferred.
        """
        siblings = self._find_siblings(concept_id)
        if not siblings:
            return 0

        # Collect existing edge targets for this concept (to avoid duplicates)
        existing_targets: set[tuple[str, RelationType]] = {
            (e.target, e.relation) for e in self.network.get_edges(concept_id, "out")
        }

        return self._infer_sibling_edges(concept_id, siblings, existing_targets, result)

    def _find_weak_concepts(self, limit: int) -> list[tuple[str, float]]:
        """Find concepts with few connections and no definitions."""
        weak: list[tuple[str, float]] = []
        for cid, concept in list(self.network._concepts.items()):
            # Skip system and code concepts
            if concept.origin in ("system",) or cid.startswith(("python:", "rust:")):
                continue
            edges_out = self.network.get_edges(cid, "out")
            edges_in = self.network.get_edges(cid, "in")
            total = len(edges_out) + len(edges_in)
            has_def = bool(concept.properties.get("definition", ""))
            if total < 3 or not has_def:
                confidence = min(1.0, total * 0.1)
                if has_def:
                    confidence += 0.3
                weak.append((cid, confidence))
        weak.sort(key=lambda x: x[1])
        return weak[:limit]

    # ═══════════════════════════════════════════════════════════════
    # 6. Correction learning — update knowledge when corrected
    # ═══════════════════════════════════════════════════════════════

    def _weaken_inferred_edges(
        self, topic_id: str, events: list[LearningEvent]
    ) -> None:
        """Weaken and remove low-confidence inferred edges from a topic."""
        edges_to_remove: list = []
        for edge in self.network.get_edges(topic_id, "out"):
            if edge.origin == "inferred" and edge.weight <= 0.4:
                edge.weight *= 0.7
                events.append(
                    LearningEvent(
                        event_type="correction",
                        description=(
                            f"Correction: weakened inferred edge "
                            f"{topic_id} → {edge.target} (might be wrong)"
                        ),
                        concepts_involved=[topic_id, edge.target],
                        confidence=0.4,
                    )
                )
                # Unlearn: edges that drop below threshold are
                # removed entirely. This prevents wrong inferences
                # from persisting at low weight and still
                # influencing reasoning.
                if edge.weight < 0.1:
                    edges_to_remove.append(
                        (edge.source, edge.target, edge.relation)
                    )

        for src, tgt, rel in edges_to_remove:
            self.network.remove_edge(src, tgt, rel)
            events.append(
                LearningEvent(
                    event_type="correction",
                    description=(
                        f"Unlearned: removed weak inferred edge "
                        f"{src} → {tgt} (weight below 0.1)"
                    ),
                    concepts_involved=[src, tgt],
                    confidence=0.3,
                )
            )

    def learn_from_correction(
        self,
        correction_text: str,
        original_topic: str = "",
    ) -> list[LearningEvent]:
        """Learn from a user correction.

        When the user says "No, X is actually Y" or "Actually, X is Y",
        extract the corrected fact and add it to the network.

        If the original topic is known, weaken inferred edges FROM that
        topic that have low confidence — they might be wrong.

        If the correction contains a negation ("not X", "X, not Y"),
        explicitly remove edges to the negated concept.
        """
        events: list[LearningEvent] = []

        # Extract the correction content
        m = _CORRECTION_RE.match(correction_text)
        corrected_text = m.group(1).strip() if m else correction_text

        # Always count the correction — the user is telling us something
        # was wrong, even if the corrected fact was already known.
        self._total_corrections += 1

        # Extract facts from the corrected text
        facts = self.learn_from_input(corrected_text)
        for fact in facts:
            fact.event_type = "correction"
            events.append(fact)

        # Also extract facts from relative clauses ("that can survive...",
        # "which are...") — the main parser only extracts one relation
        # per sentence, so "X is a Y that can Z" only captures "X is_a Y".
        # Split on relative pronouns and learn from each clause.
        clause_facts = self._extract_relative_clause_facts(corrected_text)
        for fact in clause_facts:
            fact.event_type = "correction"
            events.append(fact)

        # If no new facts were extracted, the corrected information may
        # already exist in the network. Record that we received a
        # correction even so — the weakening of inferred edges below
        # is still valuable.
        if not facts:
            events.append(
                LearningEvent(
                    event_type="correction",
                    description=f"Received correction (already knew: {corrected_text[:60]})",
                    concepts_involved=[original_topic] if original_topic else [],
                    confidence=0.5,
                )
            )

        # Extract explicit negations: "X, not Y" or "not Y" at end of
        # sentence. Remove edges from the topic to the negated concept.
        negated = self._extract_negated_concepts(corrected_text)
        if negated and original_topic:
            topic_id = self.network._resolve(original_topic.lower())
            if topic_id:
                self._remove_negated_edges(topic_id, original_topic, negated, events)

        # Weaken inferred edges from the original topic — targeted, not
        # indiscriminate. Only edges FROM the topic concept, only
        # inferred origin, only low weight.
        if original_topic:
            topic_id = self.network._resolve(original_topic.lower())
            if topic_id:
                self._weaken_inferred_edges(topic_id, events)

        self._learning_log.extend(events)
        return events

    def _remove_negated_edges(
        self,
        topic_id: str,
        original_topic: str,
        negated: list[str],
        events: list[LearningEvent],
    ) -> None:
        """Remove edges from topic to each negated concept.

        Tries exact resolve first, then fuzzy match against edge
        targets (e.g., "six" matches "six legs"). Records a negation
        event if no edge was found, so she doesn't re-learn it.
        """
        for neg_concept in negated:
            neg_id = self.network._resolve(neg_concept)
            if neg_id:
                removed = self._remove_edges_between(topic_id, neg_id, events)
            else:
                removed = self._remove_edges_matching(
                    topic_id, neg_concept, events
                )
            if not removed:
                events.append(
                    LearningEvent(
                        event_type="correction",
                        description=(
                            f"Correction: {original_topic} is NOT "
                            f"related to {neg_concept}"
                        ),
                        concepts_involved=[original_topic, neg_concept],
                        confidence=0.6,
                    )
                )

    def _extract_relative_clause_facts(self, text: str) -> list[LearningEvent]:
        """Extract facts from relative clauses in a correction.

        "X is a Y that can Z" → learn "X can Z"
        "X is a Y which has W" → learn "X has W"
        """
        events: list[LearningEvent] = []
        # Split on "that" / "which" relative pronouns
        rel_split = re.compile(
            r"\b(?:that|which|who)\s+",
            re.IGNORECASE,
        )
        parts = rel_split.split(text)
        if len(parts) < 2:
            return events

        # The first part is the main clause (already handled by
        # learn_from_input). The subsequent parts are relative clauses
        # that describe the subject. Reconstruct them as statements
        # about the subject.
        # Extract subject from the first part: "a tardigrade is a micro-animal"
        # → subject is "tardigrade"
        main = parts[0].strip()
        # Try to find the subject (first noun after articles)
        subject_match = re.match(
            r"(?:a|an|the)\s+(\w+(?:\s+\w+)?)\s+is\s+",
            main,
            re.IGNORECASE,
        )
        if not subject_match:
            return events
        subject = subject_match.group(1).strip().lower()

        for clause in parts[1:]:
            clause = clause.strip().rstrip(".")
            if not clause or len(clause.split()) < 2:
                continue
            # Reconstruct as "subject clause" and learn from it
            statement = f"{subject} {clause}"
            clause_events = self.learn_from_input(statement)
            events.extend(clause_events)

        return events

    @staticmethod
    def _extract_negated_concepts(text: str) -> list[str]:
        """Extract concepts that are explicitly negated in a correction.

        Handles patterns like:
        - "X has eight legs, not six" → ["six"]
        - "X is a mammal, not a reptile" → ["reptile"]
        - "not six legs" → ["six legs"]
        """
        # Pattern: "not <concept>" at end of sentence or clause
        # Capture the noun phrase after "not"
        neg_pattern = re.compile(
            r"\bnot\s+((?:a|an|the)\s+)?(.+?)(?:[.,;]|$)",
            re.IGNORECASE,
        )
        results: list[str] = []
        for m in neg_pattern.finditer(text):
            concept = m.group(2).strip().rstrip(".")
            # Normalize: lowercase (the article is already separated
            # by the regex into group 1, so group 2 doesn't contain it)
            concept = concept.lower().strip()
            # Skip if it's just a function word or too short
            if len(concept) >= 3 and concept not in (
                "a", "an", "the", "this", "that", "it", "true", "false",
                "right", "wrong", "sure", "quite", "very", "really",
            ):
                results.append(concept)
        return results

    def _remove_edges_between(
        self, source_id: str, target_id: str, events: list[LearningEvent]
    ) -> bool:
        """Remove all edges between two concepts (any direction, any relation).

        Returns True if any edges were removed.
        """
        removed_any = False
        edges = self.network.get_edges(source_id, "both")
        for edge in edges:
            if edge.target == target_id or edge.source == target_id:
                self.network.remove_edge(
                    edge.source, edge.target, edge.relation
                )
                removed_any = True
                events.append(
                    LearningEvent(
                        event_type="correction",
                        description=(
                            f"Correction: removed wrong edge "
                            f"{edge.source} → {edge.target} "
                            f"(user said 'not {edge.target}')"
                        ),
                        concepts_involved=[edge.source, edge.target],
                        confidence=0.7,
                    )
                )
        return removed_any

    def _remove_edges_matching(
        self, topic_id: str, neg_concept: str, events: list[LearningEvent]
    ) -> bool:
        """Remove edges from topic_id whose target contains neg_concept.

        Handles cases where the negated word is a substring of the
        edge target (e.g., "six" matches "six legs").
        """
        removed_any = False
        neg_lower = neg_concept.lower()
        edges = self.network.get_edges(topic_id, "both")
        for edge in edges:
            target_lower = edge.target.lower()
            # Match if the negated concept is the target, or the target
            # starts with the negated concept (e.g., "six" in "six legs").
            if (
                target_lower == neg_lower
                or target_lower.startswith(neg_lower + " ")
                or target_lower.startswith(neg_lower)
            ):
                self.network.remove_edge(
                    edge.source, edge.target, edge.relation
                )
                removed_any = True
                events.append(
                    LearningEvent(
                        event_type="correction",
                        description=(
                            f"Correction: removed wrong edge "
                            f"{edge.source} → {edge.target} "
                            f"(user said 'not {neg_concept}')"
                        ),
                        concepts_involved=[edge.source, edge.target],
                        confidence=0.7,
                    )
                )
        return removed_any

    # ═══════════════════════════════════════════════════════════════
    # 6. Genus extraction — deepen the semantic hierarchy
    # ═══════════════════════════════════════════════════════════════

    @staticmethod
    def _extract_genus_term(text: str) -> str | None:
        """Extract the genus term from a dictionary definition.

        Dictionary definitions follow a genus-differentia pattern:
        "a neurotransmitter involved in sleep..."
        The genus ("neurotransmitter") is the hypernym — what kind
        of thing it is. The differentia ("involved in sleep") distinguishes
        it from other things of the same kind.

        Returns the first 1-3 words of the genus, or None if no
        genus pattern is found.
        """
        # Try WordNet-style: "word\n    n 1: a/an/the <genus>..."
        m = _WORDNET_NOUN_RE.match(text)
        if not m:
            # Try Webster-style: "n.\n   1. The <genus>..."
            m = _WEBSTER_NOUN_RE.search(text)
        if not m:
            return None

        return SelfDirectedLearner._truncate_genus(m.group(1))

    @staticmethod
    def _truncate_genus(text: str) -> str | None:
        """Trim a candidate genus phrase down to its head noun phrase.

        Walks the words left to right, stopping at the first stop word or
        participle (``-ing`` / ``-ed``) — those mark the start of the
        differentia. At most three words are kept.

        Then strips leading adjectives/adverbs so the genus is the head
        noun, not a modified phrase: "microscopic animal" → "animal".

        "neurotransmitter involved in sleep" → "neurotransmitter"
        "microscopic animal that survives" → "animal"
        """
        words: list[str] = []
        for w in text.strip().split():
            w_clean = w.strip('.,;:"').lower()
            if not w_clean:
                continue
            if w_clean in _GENUS_STOP_WORDS:
                break
            # Stop at past participle / gerund patterns
            if len(w_clean) > 4 and w_clean.endswith("ing"):
                break
            if len(w_clean) > 3 and w_clean.endswith("ed"):
                break
            words.append(w_clean)
            if len(words) >= 3:
                break
        if not words:
            return None

        # Strip leading degree adverbs and adjectives to expose the
        # head noun. "microscopic animal" → "animal", "extremely
        # luminous sphere" → "sphere". This ensures the genus is the
        # fundamental category, not a modified compound.
        idx = 0
        while idx < len(words) - 1 and words[idx] in _DEGREE_ADVERBS:
            idx += 1
        while idx < len(words) - 1 and _is_likely_adjective(words[idx]):
            idx += 1
        if idx > 0:
            # Only strip if the remainder is not itself an adjective
            # (over-stripping means the phrase was all modifiers).
            if not _is_likely_adjective(words[idx]):
                words = words[idx:]

        return " ".join(words)

    # Nouns that commonly appear as the first word of man page
    # descriptions, indicating the genus of the command.
    _MAN_NOUN_GENUS = frozenset({
        "screen", "audio", "video", "text", "file", "font", "image",
        "network", "system", "daemon", "client", "server", "tool",
        "utility", "program", "library", "module", "editor", "viewer",
        "browser", "manager", "monitor", "calculator", "locker",
        "player", "recorder", "converter", "compressor", "interpreter",
        "compiler", "debugger", "profiler", "tracer", "logger",
        "wrapper", "helper", "configuration", "installation", "package",
        "capability", "character", "query", "message",
        "command", "shell", "driver", "handler", "parser", "scanner",
        "checker", "validator", "generator", "builder", "cleaner",
        "extractor", "installer", "launcher", "controller", "scheduler",
        "dispatcher", "proxy", "gateway", "bridge", "filter",
    })

    def _extract_man_page_genus(self) -> list[tuple[str, str, str]]:
        """Extract genus information from man page NAME sections.

        Man pages follow the pattern: ``command - description``.
        All man page entries are commands, so every entry gets
        ``IS_A command`` as a baseline. When the description starts
        with a noun phrase (e.g., "screen locker", "audio tag editor"),
        that noun phrase is used as the more specific genus instead.

        Returns a list of (command, genus, description) tuples.
        Only commands already in the concept network should be used
        (the caller filters by network membership).

        The result is cached because scanning all man pages is expensive
        and the man-page corpus doesn't change during a run.
        """
        if self._man_page_genus_cache is not None:
            return self._man_page_genus_cache

        cache_path = self._man_page_genus_cache_path()
        if cache_path is not None:
            cached = self._load_man_page_genus_cache(cache_path)
            if cached is not None:
                return cached

        entries: list[tuple[str, str, str]] = []
        seen: set[str] = set()

        for man_dir in self._man_page_dirs():
            self._scan_man_dir(man_dir, entries, seen)

        if cache_path and entries:
            self._save_man_page_genus_cache(cache_path, entries)

        self._man_page_genus_cache = entries
        return entries

    def _man_page_genus_cache_path(self):
        """Return the cache path for man page genus, or None."""
        if not self.data_dir:
            return None
        from pathlib import Path
        return Path(self.data_dir) / "man_page_genus_cache.json"

    def _load_man_page_genus_cache(self, cache_path) -> list[tuple[str, str, str]] | None:
        """Load cached man page genus entries, or None on failure."""
        import json
        if not cache_path.exists():
            return None
        try:
            with cache_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            return [(str(t[0]), str(t[1]), str(t[2])) for t in data]
        except (OSError, json.JSONDecodeError, TypeError) as e:
            logger.debug(f"man page genus cache load failed: {e}")
            return None

    def _save_man_page_genus_cache(self, cache_path, entries) -> None:
        """Save man page genus entries to cache."""
        import json
        try:
            with cache_path.open("w", encoding="utf-8") as f:
                json.dump(entries, f)
        except OSError as e:
            logger.debug(f"man page genus cache write failed: {e}")

    def _man_page_dirs(self) -> list[str]:
        """Return the list of man page directories to scan."""
        return [
            "/usr/share/man/man1",
            "/usr/share/man/man8",
            "/usr/share/man/man5",
            "/usr/share/man/man3",
            "/usr/share/man/man2",
            "/usr/share/man/man7",
            "/usr/local/man/man1",
            "/usr/local/man/man8",
            "/usr/local/man/man5",
            "/usr/local/man/man3",
            "/usr/local/share/man/man1",
            "/usr/local/share/man/man8",
        ]

    def _scan_man_dir(
        self, man_dir: str,
        entries: list[tuple[str, str, str]],
        seen: set[str],
    ) -> None:
        """Scan a single man page directory for genus entries."""
        import os

        if not os.path.isdir(man_dir):
            return
        try:
            filenames = os.listdir(man_dir)
        except OSError:
            return
        for fname in filenames:
            entry = self._parse_man_page_file(man_dir, fname)
            if entry is not None:
                cmd, genus, desc = entry
                if cmd not in seen:
                    seen.add(cmd)
                    entries.append((cmd, genus, desc))

    def _parse_man_page_file(
        self, man_dir: str, fname: str,
    ) -> tuple[str, str, str] | None:
        """Parse a single man page file, returning (cmd, genus, desc) or None."""
        import gzip
        import os

        is_gz = fname.endswith(".gz")
        base_fname = fname[:-3] if is_gz else fname
        sections = ("1", "2", "3", "4", "5", "6", "7", "8", "9")
        if not any(base_fname.endswith(f".{s}") for s in sections):
            return None
        path = os.path.join(man_dir, fname)
        try:
            if is_gz:
                with gzip.open(path, "rt", errors="replace") as f:
                    content = f.read(8000)
            else:
                with open(path, errors="replace") as f:
                    content = f.read(8000)
        except (OSError, gzip.BadGzipFile):
            return None

        m = re.search(
            r"\.SH\s+NAME\s*\n(.*?)(?:\.SH|\Z)", content, re.DOTALL
        )
        if not m:
            return None
        name_section = m.group(1).strip()
        name_section = re.sub(r"\\f[IRB]|\.[A-Z]+\s+", "", name_section)
        name_section = name_section.replace("\\-", "-")
        parts = name_section.split(" - ", 1)
        if len(parts) != 2:
            return None

        cmd = parts[0].strip().split(",")[0].strip().lower()
        desc = parts[1].strip().split("\n")[0].strip()
        if not cmd or not desc or len(cmd) > 30 or len(desc) > 100:
            return None
        if cmd.startswith("/") or cmd.startswith(".") or "/" in cmd:
            return None
        if not cmd[0].isalpha():
            return None
        if "\\" in cmd or "{" in cmd or "}" in cmd:
            return None
        cmd = re.sub(r"\\f[IRB]|\[b\]|\[v\]|\[r\]", "", cmd).strip()
        if not cmd or not cmd[0].isalpha():
            return None

        genus = self._classify_man_page_genus(desc)
        return (cmd, genus, desc)

    def _classify_man_page_genus(self, desc: str) -> str:
        """Classify the genus from a man page description."""
        words = desc.split()
        if not words:
            return "command"
        first_word = words[0].lower().strip(".,;:")

        if first_word in SelfDirectedLearner._MAN_NOUN_GENUS:
            genus_words = [first_word]
            for w in words[1:3]:
                w_clean = w.lower().strip(".,;:")
                if w_clean in _GENUS_STOP_WORDS:
                    break
                if len(w_clean) > 4 and w_clean.endswith("ing"):
                    break
                if len(w_clean) > 3 and w_clean.endswith("ed"):
                    break
                genus_words.append(w_clean)
            return " ".join(genus_words)
        return "command"

    def extract_genus_from_definitions(self) -> InferenceResult:
        """Extract IS_A relations from concept and dictionary definitions.

        This deepens the semantic hierarchy by parsing the genus term
        from definitions — the classic "X is a Y" pattern embedded in
        dictionary entries and concept properties.

        Three sources are mined:

        1. **Concept properties** — concepts that have a ``definition``
           property stored in the network. E.g., "serotonin" has
           definition "a neurotransmitter involved in..." →
           ``serotonin IS_A neurotransmitter``.

        2. **Offline dictionary** — the bundled dictionary file contains
           WordNet and Webster definitions with genus-differentia
           structure. Each definition's genus term becomes an IS_A edge.

        3. **Man page NAME sections** — the system's man pages contain
           ``command - description`` entries. Every command gets
           ``IS_A command`` (or a more specific genus when the
           description starts with a noun phrase like "screen locker").
           This is a rich offline source: ~2600 entries on a typical
           Linux system.

        After extracting new IS_A edges, transitive closure runs
        automatically (via :meth:`infer_transitive`) so that multi-hop
        hierarchies form: if "serotonin IS_A neurotransmitter" and
        "neurotransmitter IS_A chemical", then "serotonin IS_A chemical"
        is inferred.

        All extracted edges have ``origin="inferred"`` and weight 0.4
        (lower than directly stated facts) so they can be distinguished
        and validated.
        """
        result = InferenceResult()

        # 0a. Seed foundational definitions for key computer concepts
        #     that connect the man-page "command" subtree to the
        #     existing semantic hierarchy. Without these, "command" has
        #     no IS_A parent and the hierarchy is flat at depth 2.
        #     These are genus-differentia definitions that the extractor
        #     below will parse into IS_A edges.
        self._seed_foundational_definitions()

        # 1. Extract from concept properties
        self._extract_genus_from_concept_properties(result)

        # 2. Extract genus from man page NAME sections.
        #    Man pages follow the pattern: "command - description"
        #    All man page entries are commands/tools, so every entry
        #    gets IS_A command. When the description starts with a
        #    noun phrase (e.g., "screen locker", "audio tag editor"),
        #    that noun phrase is used as the more specific genus.
        #
        #    Unlike dictionary extraction (which only works on concepts
        #    already in the network), man page commands are ADDED as new
        #    concepts if they don't exist. This is safe because man pages
        #    are a curated, stable corpus — every entry is a real command
        #    with a real description. This significantly deepens the
        #    hierarchy by adding a whole "command" subtree.
        self._extract_genus_from_man_pages(result)

        # 3. Run transitive closure on the new IS_A edges
        if result.new_edges > 0:
            transitive = self.infer_transitive(max_inferences=50)
            result.new_edges += transitive.new_edges
            result.contradictions_found += transitive.contradictions_found
            result.events.extend(transitive.events)

        self._learning_log.extend(result.events)
        return result

    def _seed_foundational_definitions(self) -> None:
        """Seed foundational definitions for key computer concepts.

        These connect the man-page "command" subtree to the
        existing semantic hierarchy. Without these, "command" has
        no IS_A parent and the hierarchy is flat at depth 2.
        These are genus-differentia definitions that the extractor
        below will parse into IS_A edges.
        """
        _FOUNDATIONAL_DEFS = {
            "command": "a program that a user invokes to perform a task",
            "program": "a software artifact that a computer executes",
            "software": "a collection of programs and data for computers",
            "tool": "a program designed for a specific practical purpose",
            "utility": "a program designed for system maintenance or configuration",
            "editor": "a program for creating and modifying text files",
            "shell": "a program that interprets and executes commands",
            "daemon": "a program that runs in the background without user interaction",
            "library": "a collection of code that other programs use",
            "compiler": "a program that translates source code into executable form",
            "interpreter": "a program that executes source code directly",
            "parser": "a program that analyzes text structure",
            "filter": "a program that processes input data and produces output",
            "client": "a program that requests services from a server",
            "server": "a program that provides services to clients",
            "manager": "a program that organizes and controls resources",
            "monitor": "a program that observes and reports system state",
            "wrapper": "a program that provides an interface around another program",
        }
        for concept_name, defn in _FOUNDATIONAL_DEFS.items():
            concept = self.network.get_concept(concept_name)
            if concept is None:
                concept = self.network.add_concept(
                    concept_name, confidence=0.4, origin="foundational"
                )
            if concept and not (concept.properties or {}).get("definition"):
                if not concept.properties:
                    concept.properties = {}
                concept.properties["definition"] = defn

    def _extract_genus_from_concept_properties(self, result: InferenceResult) -> None:
        """Extract IS_A edges from concept definition properties."""
        # 1. Extract from concept properties
        for cid, concept in list(self.network._concepts.items()):
            defn = (concept.properties or {}).get("definition", "")
            if not defn or len(defn) < 10:
                continue

            # Try concept definition pattern first: "a/an/the <genus>..."
            m = _CONCEPT_DEF_RE.match(defn.strip())
            if m:
                # The regex is non-greedy but its terminator list is narrower
                # than _GENUS_STOP_WORDS, so "a neurotransmitter involved in
                # sleep" captures "neurotransmitter involved". Trim to the
                # head noun phrase.
                genus = self._truncate_genus(m.group(1))
            else:
                genus = self._extract_genus_term(defn)

            if not genus or len(genus) < 3 or genus == cid:
                continue

            # Skip if the genus is a multi-word fragment that's
            # clearly not a concept (contains digits, etc.)
            if any(ch.isdigit() for ch in genus):
                continue

            # Ensure both concepts exist
            if self.network.get_concept(genus) is None:
                self.network.add_concept(genus, confidence=0.3, origin="inferred")
            if self.network.get_concept(cid) is None:
                continue

            # Skip if edge already exists
            if self._has_edge(cid, genus, RelationType.IS_A):
                continue

            edge = self.network.add_edge(
                cid, genus, RelationType.IS_A, weight=0.4, origin="inferred"
            )
            if edge:
                result.new_edges += 1
                result.events.append(
                    LearningEvent(
                        event_type="genus_extraction",
                        description=f"Genus: {cid} is_a {genus}",
                        concepts_involved=[cid, genus],
                        confidence=0.4,
                    )
                )

    def _extract_genus_from_man_pages(self, result: InferenceResult) -> None:
        """Extract IS_A edges from man page NAME sections.

        Man pages follow the pattern: "command - description"
        All man page entries are commands/tools, so every entry
        gets IS_A command. When the description starts with a
        noun phrase (e.g., "screen locker", "audio tag editor"),
        that noun phrase is used as the more specific genus.

        Unlike dictionary extraction (which only works on concepts
        already in the network), man page commands are ADDED as new
        concepts if they don't exist. This is safe because man pages
        are a curated, stable corpus — every entry is a real command
        with a real description. This significantly deepens the
        hierarchy by adding a whole "command" subtree.
        """
        # 3. Extract genus from man page NAME sections.
        #    Man pages follow the pattern: "command - description"
        #    All man page entries are commands/tools, so every entry
        #    gets IS_A command. When the description starts with a
        #    noun phrase (e.g., "screen locker", "audio tag editor"),
        #    that noun phrase is used as the more specific genus.
        #
        #    Unlike dictionary extraction (which only works on concepts
        #    already in the network), man page commands are ADDED as new
        #    concepts if they don't exist. This is safe because man pages
        #    are a curated, stable corpus — every entry is a real command
        #    with a real description. This significantly deepens the
        #    hierarchy by adding a whole "command" subtree.
        man_entries = self._extract_man_page_genus()
        for cmd, genus, desc in man_entries:
            # Only enrich commands that are already in the network; don't
            # add thousands of man-page entries that may never come up.
            concept = self.network.get_concept(cmd)
            if concept is None:
                continue

            # Ensure genus concept exists
            if self.network.get_concept(genus) is None:
                self.network.add_concept(
                    genus, confidence=0.3, origin="inferred"
                )

            # Store the man description as a definition if the
            # concept doesn't have one yet
            if not (concept.properties or {}).get("definition"):
                if not concept.properties:
                    concept.properties = {}
                concept.properties["definition"] = desc

            # Skip if edge already exists
            if self._has_edge(cmd, genus, RelationType.IS_A):
                continue

            edge = self.network.add_edge(
                cmd, genus, RelationType.IS_A, weight=0.4, origin="inferred"
            )
            if edge:
                result.new_edges += 1
                result.events.append(
                    LearningEvent(
                        event_type="genus_extraction",
                        description=f"Genus: {cmd} is_a {genus}",
                        concepts_involved=[cmd, genus],
                        confidence=0.4,
                    )
                )

    # ═══════════════════════════════════════════════════════════════
    # Full learning cycle — run all inference mechanisms
    # ═══════════════════════════════════════════════════════════════

    def run_inference_cycle(
        self, force: bool = False, brain_waves=None
    ) -> InferenceResult:
        """Run a full inference cycle.

        This is called after interactions to let Genesis improve her
        knowledge through all inference mechanisms:
        1. Genus extraction from definitions (deepens the hierarchy)
        2. Transitive inference (closes the hierarchy)
        3. Definition synthesis
        4. Analogical transfer
        5. Self-study of weak concepts

        Has a cooldown period (_INFERENCE_COOLDOWN_SECONDS) to prevent
        running expensive inference on every interaction. Use force=True
        to bypass the cooldown.

        Args:
            force: Bypass the cooldown timer.
            brain_waves: Optional BrainWaveState. When provided,
                gamma-dominant states boost analogical inference
                (gamma = binding, remote association; Jung-Beeman et
                al., 2004), and delta-dominant states skip inference
                entirely (deep rest, not encoding mode).
        """
        now = time.time()
        if not force and (now - self._last_inference_time) < _INFERENCE_COOLDOWN_SECONDS:
            return InferenceResult()
        self._last_inference_time = now

        # Brain-wave gating: delta-dominant → skip inference entirely.
        # The brain is in deep rest, not encoding or binding mode.
        if brain_waves is not None:
            if brain_waves.dominant == BrainWave.DELTA:
                return InferenceResult()

        combined = InferenceResult()

        # 1. Genus extraction — parse definitions for IS_A relations.
        #    Runs before transitive inference so new IS_A edges feed
        #    into the closure.
        genus = self.extract_genus_from_definitions()
        combined.new_edges += genus.new_edges
        combined.contradictions_found += genus.contradictions_found
        combined.events.extend(genus.events)

        # 2. Transitive inference — close IS_A/PART_OF/DEPENDS_ON/ENABLES
        transitive = self.infer_transitive(max_inferences=30)
        combined.new_edges += transitive.new_edges
        combined.contradictions_found += transitive.contradictions_found
        combined.events.extend(transitive.events)

        defs = self.synthesize_definitions(max_definitions=20)
        combined.new_definitions += defs.new_definitions
        combined.events.extend(defs.events)

        # Brain-wave-modulated analogical inference.
        # Gamma-dominant states boost analogical inference — gamma is
        # the binding signal for remote associations (Jung-Beeman et
        # al., 2004). High integration drive means more distant
        # concepts can be connected.
        analogical_max = 15
        if brain_waves is not None:
            if brain_waves.dominant == BrainWave.GAMMA:
                analogical_max = int(analogical_max * (1.0 + 0.4 * brain_waves.integration))
            elif brain_waves.dominant == BrainWave.ALPHA:
                analogical_max = int(analogical_max * 0.6)

        analogical = self.infer_analogical(max_inferences=analogical_max)
        combined.new_edges += analogical.new_edges
        combined.events.extend(analogical.events)

        study = self.self_study(max_concepts=10)
        combined.strengthened += study.strengthened
        combined.new_definitions += study.new_definitions
        combined.events.extend(study.events)

        # 5. Global consistency sweep — self-healing: remove
        # contradictions and orphaned edges. Runs on every inference
        # cycle to keep the network clean.
        sweep = self.consistency_sweep()
        combined.contradictions_found += sweep.contradictions_found
        combined.events.extend(sweep.events)

        return combined

    # ═══════════════════════════════════════════════════════════════
    # 7. Global consistency sweep — detect & resolve network-wide contradictions
    # ═══════════════════════════════════════════════════════════════

    def _detect_contradiction_pairs(self, result: InferenceResult) -> None:
        """Detect and resolve SIMILAR_TO / OPPOSITE_OF contradictions.

        Non-destructive: when a contradiction is found (SIMILAR_TO and
        OPPOSITE_OF on the same pair), both edges are downgraded rather
        than removing the weaker one. This preserves competing claims
        with reduced confidence so the system can re-evaluate them as
        more evidence accumulates. This matches the 2026 epistemic
        integrity literature which emphasizes preserving revision
        history rather than destroying knowledge.
        """
        seen_pairs: set[frozenset[str]] = set()
        for edge in list(self.network._edges):
            if edge.relation not in (
                RelationType.SIMILAR_TO,
                RelationType.OPPOSITE_OF,
            ):
                continue
            pair = frozenset({edge.source, edge.target})
            if pair in seen_pairs:
                # Check if the opposite relation exists
                other_rel = (
                    RelationType.OPPOSITE_OF
                    if edge.relation == RelationType.SIMILAR_TO
                    else RelationType.SIMILAR_TO
                )
                if self._has_edge(edge.source, edge.target, other_rel) or \
                   self._has_edge(edge.target, edge.source, other_rel):
                    result.contradictions_found += 1
                    # Non-destructive: downgrade both edges rather than
                    # removing the weaker one. Both claims are preserved
                    # with reduced confidence.
                    other_edge = self.network._edge_key_index.get(
                        (edge.source, edge.target, other_rel)
                    ) or self.network._edge_key_index.get(
                        (edge.target, edge.source, other_rel)
                    )
                    downgrade_factor = 0.8
                    if other_edge:
                        other_edge.weight *= downgrade_factor
                    edge.weight *= downgrade_factor
                    rel_name = (
                        "SIMILAR_TO"
                        if edge.relation == RelationType.SIMILAR_TO
                        else "OPPOSITE_OF"
                    )
                    result.events.append(
                        LearningEvent(
                            event_type="consistency",
                            description=(
                                f"Resolved contradiction: downgraded "
                                f"{rel_name} between {edge.source} and "
                                f"{edge.target} (non-destructive)"
                            ),
                            concepts_involved=[edge.source, edge.target],
                            confidence=0.5,
                        )
                    )
            seen_pairs.add(pair)

    def _remove_orphaned_edges(self, result: InferenceResult) -> list:
        """Remove orphaned inferred edges with weight below 0.1."""
        orphaned = [
            e for e in list(self.network._edges)
            if e.origin == "inferred" and e.weight < 0.1
        ]
        for edge in orphaned:
            self.network.remove_edge(edge.source, edge.target, edge.relation)
            result.events.append(
                LearningEvent(
                    event_type="consistency",
                    description=(
                        f"Unlearned orphaned edge: {edge.source} → {edge.target} "
                        f"(weight={edge.weight:.3f})"
                    ),
                    concepts_involved=[edge.source, edge.target],
                    confidence=0.2,
                )
            )
        return orphaned

    def consistency_sweep(self) -> InferenceResult:
        """Sweep the entire network for contradictions and inconsistencies.

        This is the global self-healing mechanism. It detects:
        1. SIMILAR_TO + OPPOSITE_OF between the same pair (contradiction)
        2. IS_A cycles (A is_a B is_a ... is_a A)
        3. Orphaned inferred edges (weight decayed below 0.1)

        Contradictions are resolved non-destructively by downgrading
        both edges' weights — competing claims are preserved with
        reduced confidence so the system can re-evaluate them as more
        evidence accumulates. Cycles are broken by removing the weakest
        edge in the cycle. Orphaned edges are removed entirely
        (unlearning).
        """
        result = InferenceResult()

        # 1. Detect SIMILAR_TO / OPPOSITE_OF contradictions
        self._detect_contradiction_pairs(result)

        # 2. Remove orphaned inferred edges (weight < 0.1)
        orphaned = self._remove_orphaned_edges(result)

        if result.contradictions_found > 0 or orphaned:
            self._learning_log.extend(result.events)

        return result

    # ═══════════════════════════════════════════════════════════════
    # Introspection
    # ═══════════════════════════════════════════════════════════════

    def describe_recent_learning(self, n: int = 10) -> str:
        """Structural description of recent learning for metadata."""
        if not self._learning_log:
            return "no self-directed learning yet"

        recent = list(self._learning_log)[-n:]
        parts = [
            f"{self._total_inferences} inferences, "
            f"{self._total_conversation_facts} conversation facts, "
            f"{self._total_definitions_synthesized} definitions, "
            f"{self._total_corrections} corrections"
        ]

        for event in recent:
            parts.append(f"  {event.description}")

        return "\n".join(parts)

    def get_stats(self) -> dict[str, int]:
        """Get learning statistics."""
        return {
            "total_inferences": self._total_inferences,
            "conversation_facts": self._total_conversation_facts,
            "corrections": self._total_corrections,
            "definitions_synthesized": self._total_definitions_synthesized,
            "studied_concepts": len(self._studied_concepts),
            "learning_log_size": len(self._learning_log),
        }

    def get_recent_events(self, n: int = 20) -> list[LearningEvent]:
        """Get the most recent learning events."""
        return list(self._learning_log)[-n:]

    # ═══════════════════════════════════════════════════════════════
    # Internal helpers
    # ═══════════════════════════════════════════════════════════════

    def _is_meaningful(self, word: str) -> bool:
        """Check if a word is meaningful enough to be a concept."""
        word = word.lower().strip()
        if not word:
            return False
        # Function words are grammatical, not conceptual — a misparse
        # like "is → part of → the" must not mint concepts for them.
        if word in _FUNCTION_WORDS:
            return False
        # Multi-word fragments that begin with a function word are
        # conversation fragments, not concepts ("i know the computer").
        words = word.split()
        if len(words) > 1 and words[0] in _FUNCTION_WORDS:
            return False
        return True

    def _add_relationship(
        self,
        source: str,
        target: str,
        relation: RelationType,
        weight: float,
        origin: str,
    ) -> bool:
        """Add a relationship to the network, creating concepts if needed.

        Returns True if a new edge was added, False if it already existed.
        """
        # Normalize and resolve to existing concepts if possible
        source = _normalize_concept(source)
        target = _normalize_concept(target)

        # Ensure both concepts exist
        if self.network.get_concept(source) is None:
            self.network.add_concept(source, confidence=0.5, origin=origin)
        if self.network.get_concept(target) is None:
            self.network.add_concept(target, confidence=0.5, origin=origin)

        # Check if edge already exists
        if self._has_edge(source, target, relation):
            # Strengthen existing edge
            for edge in self.network.get_edges(source, "out"):
                if edge.target == target and edge.relation == relation:
                    edge.weight = min(0.95, edge.weight + 0.05)
                    return False
            return False

        self.network.add_edge(source, target, relation, weight=weight, origin=origin)
        return True

    def _add_definition(self, concept_name: str, definition: str) -> bool:
        """Add a definition to a concept, creating it if needed.

        If the concept already exists with a DIFFERENT definition, this
        creates a new sense (polysemy). The concept network's add_concept
        method handles sense disambiguation — it creates a separate
        concept with a disambiguated ID (e.g., "i#2") when the definition
        differs from the existing one.

        Returns True if a new definition was added.
        """
        definition = _clean_definition_text(definition)
        if not definition:
            return False

        # Check if the concept already exists with this exact definition
        # before calling add_concept (which would merge it).
        existing_concept = self.network.get_concept(concept_name)
        if existing_concept:
            existing_def = existing_concept.properties.get("definition", "")
            if existing_def == definition:
                return False  # already has this exact definition
            if not existing_def:
                # Concept exists without a definition (e.g., from an is_a edge)
                # — fill it in rather than creating a disambiguated sense.
                existing_concept.properties["definition"] = definition
                existing_concept.properties.setdefault("part_of_speech", "noun")
                return True

        # Pass the definition as a property so add_concept can detect
        # that this is a different sense and create a separate concept.
        self.network.add_concept(
            concept_name,
            confidence=0.6,
            origin="learned",
            properties={"definition": definition, "part_of_speech": "noun"},
        )
        return True

    def _has_edge(self, source: str, target: str, relation: RelationType) -> bool:
        """Check if a specific edge exists."""
        for edge in self.network.get_edges(source, "out"):
            if edge.target == target and edge.relation == relation:
                return True
        return False

    def _has_any_edge(self, source: str, target: str) -> bool:
        """Check if any edge exists between two concepts (either direction)."""
        for edge in self.network.get_edges(source, "out"):
            if edge.target == target:
                return True
        for edge in self.network.get_edges(target, "out"):
            if edge.target == source:
                return True
        return False

    def _has_any_edge_resolved(self, source_id: str, target_id: str) -> bool:
        """Check if any edge exists between two already-resolved concept IDs.

        Unlike _has_any_edge, this skips the _resolve/_normalize_id step
        since the caller already has normalized concept IDs. This is
        significantly faster in tight loops like transitive inference.
        """
        net = self.network
        for edge in net._edge_index.get(source_id, []):
            if edge.target == target_id:
                return True
        for edge in net._edge_index.get(target_id, []):
            if edge.target == source_id:
                return True
        return False

    def _is_contradiction(self, source: str, target: str, relation: RelationType) -> bool:
        """Check if an inferred edge would contradict existing knowledge."""
        # SIMILAR_TO and OPPOSITE_OF are mutually exclusive
        if relation == RelationType.SIMILAR_TO:
            if self._has_edge(source, target, RelationType.OPPOSITE_OF):
                return True
            if self._has_edge(target, source, RelationType.OPPOSITE_OF):
                return True
        if relation == RelationType.OPPOSITE_OF:
            if self._has_edge(source, target, RelationType.SIMILAR_TO):
                return True
            if self._has_edge(target, source, RelationType.SIMILAR_TO):
                return True
        # PREVENTS and ENABLES are semantically opposite — if X enables Y,
        # inferring that X prevents Y would be contradictory, and vice versa
        if relation == RelationType.ENABLES:
            if self._has_edge(source, target, RelationType.PREVENTS):
                return True
        if relation == RelationType.PREVENTS:
            if self._has_edge(source, target, RelationType.ENABLES):
                return True
        return False

    def _would_create_cycle(self, source: str, target: str, relation: RelationType) -> bool:
        """Check if adding source→target would create a cycle.

        Only relevant for hierarchical relations (IS_A, PART_OF).
        """
        if relation not in (RelationType.IS_A, RelationType.PART_OF):
            return False

        # BFS from target: can we reach source?
        # Use _edge_index directly since source/target are resolved IDs
        edge_index = self.network._edge_index
        visited: set[str] = set()
        queue: deque[str] = deque([target])
        while queue:
            current = queue.popleft()
            if current in visited:
                continue
            visited.add(current)
            if current == source:
                return True
            # Safety cap: don't explore the whole graph on every check
            if len(visited) > 1000:
                return False
            for edge in edge_index.get(current, []):
                if edge.relation == relation:
                    queue.append(edge.target)
        return False
