"""Concept classification — category/modality detection, quality filters."""

from __future__ import annotations

import re

from .types import ConceptCategory, ConceptModality, Edge, RelationType

_WORD_RE = re.compile(r"[A-Za-z']+")
_SENTENCE_SPLIT_RE = re.compile(r"[.!?]+")
_CAMEL_SPLIT_RE = re.compile(r"(?<=[a-z])(?=[A-Z])|[_:.]+")
_CLAUSE_SPLIT_RE = re.compile(r"\s*;\s*")
_COMPOUND_SPLIT_RE = re.compile(r"\s*,\s*(?:and\s+)?|\s+and\s+")
_RELATIVE_CLAUSE_RE = re.compile(
    r"\b(which|who|that|where|when|why|whose|whom)\s+",
    re.IGNORECASE,
)
_RELATION_VERBS: frozenset[str] = frozenset({
    "is", "are", "was", "were",
    "cause", "causes", "caused", "causing",
    "lead", "leads", "leaded", "leading",
    "emerge", "emerges", "emerged", "emerging",
    "arise", "arises", "arose", "arising", "stem", "stems", "stemmed", "stemming",
    "originate", "originates", "originated", "originating",
    "depend", "depends", "depended", "depending",
    "require", "requires", "required", "requiring",
    "rely", "relies", "relied", "relying",
    "need", "needs", "needed", "needing",
    "enable", "enables", "enabled", "enabling",
    "support", "supports", "supported", "supporting",
    "enhance", "enhances", "enhanced", "enhancing",
    "strengthen", "strengthens", "strengthened", "strengthening",
    "facilitate", "facilitates", "facilitated", "facilitating",
    "promote", "promotes", "promoted", "promoting",
    "prevent", "prevents", "prevented", "preventing",
    "block", "blocks", "blocked", "blocking",
    "suppress", "suppresses", "suppressed", "suppressing",
    "inhibit", "inhibits", "inhibited", "inhibiting",
    "excite", "excites", "excited", "exciting",
    "activate", "activates", "activated", "activating",
    "deactivate", "deactivates", "deactivated", "deactivating",
    "stimulate", "stimulates", "stimulated", "stimulating",
    "create", "creates", "created", "creating",
    "form", "forms", "formed", "forming",
    "generate", "generates", "generated", "generating",
    "produce", "produces", "produced", "producing",
    "trigger", "triggers", "triggered", "triggering",
    "induce", "induces", "induced", "inducing",
    "provoke", "provokes", "provoked", "provoking",
    "harm", "harms", "harmed", "harming",
    "impair", "impairs", "impaired", "impairing",
    "damage", "damages", "damaged", "damaging",
    "degrade", "degrades", "degraded", "degrading",
    "disrupt", "disrupts", "disrupted", "disrupting",
    "resemble", "resembles", "resembled", "resembling",
    "contradict", "contradicts", "contradicted", "contradicting",
    "restore", "restores", "restored", "restoring",
    "regulate", "regulates", "regulated", "regulating",
    "control", "controls", "controlled", "controlling",
    "modulate", "modulates", "modulated", "modulating",
    "increase", "increases", "increased", "increasing",
    "decrease", "decreases", "decreased", "decreasing",
    "reduce", "reduces", "reduced", "reducing",
    "elevate", "elevates", "elevated", "elevating",
    "lower", "lowers", "lowered", "lowering",
    "improve", "improves", "improved", "improving",
    "worsen", "worsens", "worsened", "worsening",
    "affect", "affects", "affected", "affecting",
    "influence", "influences", "influenced", "influencing",
    "determine", "determines", "determined", "determining",
    "mediate", "mediates", "mediated", "mediating",
    "underlie", "underlies", "underlay", "underlying",
    "subserve", "subservd", "subserving",
    "sustain", "sustains", "sustained", "sustaining",
    "permit", "permits", "permitted", "permitting",
    "allow", "allows", "allowed", "allowing",
    "protect", "protects", "protected", "protecting",
    "shield", "shields", "shielded", "shielding",
    "defend", "defends", "defended", "defending",
    "clear", "clears", "cleared", "clearing",
    "remove", "removes", "removed", "removing",
    "rebuild", "rebuilds", "rebuilt", "rebuilding",
    "include", "includes", "included", "including",
    "involve", "involves", "involved", "involving",
    "contain", "contains", "contained", "containing",
    "consist", "consists", "consisted", "consisting",
    "compose", "composes", "composed", "composing",
    "comprise", "comprises", "comprised", "comprising",
    "constitute", "constitutes", "constituted", "constituting",
    "represent", "represents", "represented", "representing",
    "denote", "denotes", "denoted", "denoting",
    "describe", "describes", "described", "describing",
    "define", "defines", "defined", "defining",
    "refer", "refers", "referred", "referring",
    "indicate", "indicates", "indicated", "indicating",
    "signify", "signifies", "signified", "signifying",
    "mean", "means", "meant", "meaning",
    "express", "expresses", "expressed", "expressing",
    "encode", "encodes", "encoded", "encoding",
    "store", "stores", "stored", "storing",
    "retrieve", "retrieves", "retrieved", "retrieving",
    "process", "processes", "processed", "processing",
    "compute", "computes", "computed", "computing",
    "calculate", "calculates", "calculated", "calculating",
    "measure", "measures", "measured", "measuring",
    "detect", "detects", "detected", "detecting",
    "sense", "senses", "sensed", "sensing",
    "perceive", "perceives", "perceived", "perceiving",
    "recognize", "recognizes", "recognized", "recognizing",
    "identify", "identifies", "identified", "identifying",
    "classify", "classifies", "classified", "classifying",
    "organize", "organizes", "organized", "organizing",
    "arrange", "arranges", "arranged", "arranging",
    "structure", "structures", "structured", "structuring",
    "function", "functions", "functioned", "functioning",
    "operate", "operates", "operated", "operating",
    "act", "acts", "acted", "acting",
    "behave", "behaves", "behaved", "behaving",
    "perform", "performs", "performed", "performing",
    "execute", "executes", "executed", "executing",
    "achieve", "achieves", "achieved", "achieving",
    "accomplish", "accomplishes", "accomplished", "accomplishing",
    "complete", "completes", "completed", "completing",
    "finish", "finishes", "finished", "finishing",
    "occur", "occurs", "occurred", "occurring",
    "happen", "happens", "happened", "happening",
    "arisen", "appear", "appears", "appeared", "appearing",
    "develop", "develops", "developed", "developing",
    "change", "changes", "changed", "changing",
    "vary", "varies", "varied", "varying",
    "differ", "differs", "differed", "differing",
    "match", "matches", "matched", "matching",
    "correspond", "corresponds", "corresponded", "corresponding",
    "compare", "compares", "compared", "comparing",
    "exceed", "exceeds", "exceeded", "exceeding",
    "dominate", "dominates", "dominated", "dominating",
    "manage", "manages", "managed", "managing",
    "direct", "directs", "directed", "directing",
    "guide", "guides", "guided", "guiding",
    "maintain", "maintains", "maintained", "maintaining",
    "communicate", "communicates", "communicated", "communicating",
    "signal", "signals", "signaled", "signalled", "signaling", "signalling",
    "transmit", "transmits", "transmitted", "transmitting",
    "propagate", "propagates", "propagated", "propagating",
    "spread", "spreads", "spreading",
    "diffuse", "diffuses", "diffused", "diffusing",
    "transfer", "transfers", "transferred", "transferring",
    "transport", "transports", "transported", "transporting",
    "release", "releases", "released", "releasing",
    "bind", "binds", "bound", "binding",
    "link", "links", "linked", "linking",
    "join", "joins", "joined", "joining",
    "attach", "attaches", "attached", "attaching",
    "combine", "combines", "combined", "combining",
    "merge", "merges", "merged", "merging",
    "separate", "separates", "separated", "separating",
    "divide", "divides", "divided", "dividing",
    "split", "splits", "splitting",
    "share", "shares", "shared", "sharing",
    "exchange", "exchanges", "exchanged", "exchanging",
    "replace", "replaces", "replaced", "replacing",
    "respond", "responds", "responded", "responding",
    "react", "reacts", "reacted", "reacting",
    "interact", "interacts", "interacted", "interacting",
    "modify", "modifies", "modified", "modifying",
    "alter", "alters", "altered", "altering",
    "adapt", "adapts", "adapted", "adapting",
    "adjust", "adjusts", "adjusted", "adjusting",
    "transform", "transforms", "transformed", "transforming",
    "convert", "converts", "converted", "converting",
    "mutate", "mutates", "mutated", "mutating",
    "evolve", "evolves", "evolved", "evolving",
    "circulate", "circulates", "circulated", "circulating",
    "flow", "flows", "flowed", "flowing",
    "cycle", "cycles", "cycled", "cycling",
    "oscillate", "oscillates", "oscillated", "oscillating",
    "fluctuate", "fluctuates", "fluctuated", "fluctuating",
    "migrate", "migrates", "migrated", "migrating",
    "grow", "grows", "grew", "grown", "growing",
    "shrink", "shrinks", "shrank", "shrunk", "shrinking",
    "expand", "expands", "expanded", "expanding",
    "contract", "contracts", "contracted", "contracting",
    "extend", "extends", "extended", "extending",
    "open", "opens", "opened", "opening",
    "close", "closes", "closed", "closing",
    "break", "breaks", "broke", "broken", "breaking",
    "fall", "falls", "fell", "fallen", "falling",
    "rise", "rises", "rose", "risen", "rising",
    "drop", "drops", "dropped", "dropping",
    "run", "runs", "ran", "running",
    "begin", "begins", "began", "begun", "beginning",
    "continue", "continues", "continued", "continuing",
    "remain", "remains", "remained", "remaining",
    "stay", "stays", "stayed", "staying",
    "keep", "keeps", "kept", "keeping",
    "hold", "holds", "held", "holding",
    "bring", "brings", "brought", "bringing",
    "carry", "carries", "carried", "carrying",
    "send", "sends", "sent", "sending",
    "receive", "receives", "received", "receiving",
    "obtain", "obtains", "obtained", "obtaining",
    "acquire", "acquires", "acquired", "acquiring",
    "gain", "gains", "gained", "gaining",
    "lose", "loses", "lost", "losing",
    "pay", "pays", "paid", "paying",
    "buy", "buys", "bought", "buying",
    "sell", "sells", "sold", "selling",
    "spend", "spends", "spent", "spending",
    "meet", "meets", "met", "meeting",
    "leave", "leaves", "left", "leaving",
    "enter", "enters", "entered", "entering",
    "reach", "reaches", "reached", "reaching",
    "approach", "approaches", "approached", "approaching",
    "pass", "passes", "passed", "passing",
    "cover", "covers", "covered", "covering",
    "fill", "fills", "filled", "filling",
    "build", "builds", "built", "building",
    "construct", "constructs", "constructed", "constructing",
    "repair", "repairs", "repaired", "repairing",
    "fix", "fixes", "fixed", "fixing",
    "throw", "throws", "threw", "thrown", "throwing",
    "catch", "catches", "caught", "catching",
    "hit", "hits", "hitting",
    "touch", "touches", "touched", "touching",
    "taste", "tastes", "tasted", "tasting",
    "smell", "smells", "smelled", "smelt", "smelling",
    "sound", "sounds", "sounded", "sounding",
    "seem", "seems", "seemed", "seeming",
    "become", "becomes", "became", "becoming",
    "try", "tries", "tried", "trying",
    "attempt", "attempts", "attempted", "attempting",
    "fail", "fails", "failed", "failing",
    "succeed", "succeeds", "succeeded", "succeeding",
    "help", "helps", "helped", "helping",
    "assist", "assists", "assisted", "assisting",
    "aid", "aids", "aided", "aiding",
    "serve", "serves", "served", "serving",
    "work", "works", "worked", "working",
    "play", "plays", "played", "playing",
    "rest", "rests", "rested", "resting",
    "sleep", "sleeps", "slept", "sleeping",
    "wake", "wakes", "woke", "woken", "waking",
    "feed", "feeds", "fed", "feeding",
    "eat", "eats", "ate", "eaten", "eating",
    "drink", "drinks", "drank", "drunk", "drinking",
    "breathe", "breathes", "breathed", "breathing",
    "exhale", "exhales", "exhaled", "exhaling",
    "inhale", "inhales", "inhaled", "inhaling",
    "secrete", "secretes", "secreted", "secreting",
    "absorb", "absorbs", "absorbed", "absorbing",
    "excrete", "excretes", "excreted", "excreting",
    "digest", "digests", "digested", "digesting",
    "metabolize", "metabolizes", "metabolized", "metabolizing",
    "synthesize", "synthesizes", "synthesized", "synthesizing",
    "depolarize", "depolarizes", "depolarized", "depolarizing",
    "repolarize", "repolarizes", "repolarized", "repolarizing",
    "hyperpolarize", "hyperpolarizes", "hyperpolarized", "hyperpolarizing",
    # Common verbs that were missing and produced phrase fragments
    # like "antiquary tended", "results showed", "derived from"
    "tend", "tends", "tended", "tending",
    "result", "results", "resulted", "resulting",
    "derive", "derives", "derived", "deriving",
    "show", "shows", "showed", "shown", "showing",
    "demonstrate", "demonstrates", "demonstrated", "demonstrating",
    "reveal", "reveals", "revealed", "revealing",
    "suggest", "suggests", "suggested", "suggesting",
    "yield", "yields", "yielded", "yielding",
    "exhibit", "exhibits", "exhibited", "exhibiting",
    "display", "displays", "displayed", "displaying",
    "characterize", "characterizes", "characterized", "characterizing",
})
_INDEPENDENT_CLAUSE_RE = re.compile(
    r"^(?:[A-Za-z][A-Za-z0-9_\-]*(?:\s+[A-Za-z][A-Za-z0-9_\-]*)*)\s+"
    r"(?:" + "|".join(re.escape(v) for v in _RELATION_VERBS) + r")\b",
    re.IGNORECASE,
)
_CLAUSE_SUBJECT_RE = re.compile(
    r"^\s*(.+?)\s+(?:" + "|".join(re.escape(v) for v in _RELATION_VERBS) + r")\b",
    re.IGNORECASE,
)
def _column_of(origin: str, concept_id: str) -> str:
    """Determine which cortical column a concept belongs to.

    Columns are determined by origin (the 'molecular gradient' that
    specifies cortical placement during development). Concepts with
    python:/rust: prefixes are in the code column regardless of their
    origin field, since the origin field may have been set by a
    different system first (e.g., conversation learning ran before
    code learning).

    The 'introspection' and 'identity' origins are merged into a single
    'identity' column. In the brain, self-knowledge (who I am) and
    self-awareness (how I think) are not separated into different cortical
    areas — they both involve midline prefrontal structures (medial PFC,
    anterior cingulate, posterior cingulate). Keeping them in one column
    is more biologically accurate.
    """
    if concept_id.startswith("python:") or concept_id.startswith("rust:"):
        return "code"
    if concept_id.startswith("identity:"):
        return "identity"
    if origin == "introspection":
        return "identity"  # merge introspection into identity column
    return origin
def _infer_concept_origin(concept_id: str) -> str:
    """Infer the appropriate origin for a concept based on its ID prefix.

    Used when ``add_edge`` auto-creates a missing concept. Without this,
    all auto-created concepts would get ``origin="conversation"`` (the
    default), which mislabels code concepts (python:/rust:) as
    conversation-origin. This caused 2266 concepts to have the wrong
    origin in the saved state.
    """
    if concept_id.startswith("python:") or concept_id.startswith("rust:"):
        return "code"
    if concept_id.startswith("identity:"):
        return "identity"
    return "conversation"
def _strip_prefix(cid: str, prefixes: tuple[str, ...] = ("python:", "rust:")) -> str:
    """Strip a known prefix from a concept ID, if present."""
    for prefix in prefixes:
        if cid.startswith(prefix):
            return cid[len(prefix) :]
    return cid
_SENSE_SUFFIX_RE = re.compile(r"#\d+$")
def strip_sense_suffix(concept_id: str) -> str:
    """Strip the polysemy sense suffix from a concept ID.

    Concept IDs for additional senses use a ``#N`` suffix (e.g.
    ``"orange#2"``).  When a concept ID is used as a lookup key for an
    external system (dictionary, man pages, web search) the suffix is
    meaningless — those systems know nothing about sense
    disambiguation.  This helper restores the bare word so that
    ``"orange#2"`` becomes ``"orange"``.
    """
    return _SENSE_SUFFIX_RE.sub("", concept_id)
_WHITESPACE_RE = re.compile(r"\s+")
_FUNCTION_WORDS: frozenset[str] = frozenset({
    # Question words
    "what", "who", "where", "when", "why", "how", "which", "whose",
    # Personal/demonstrative pronouns
    "i", "me", "my", "mine", "we", "us", "our", "ours",
    "you", "your", "yours", "he", "him", "his", "she", "her", "hers",
    "it", "its", "they", "them", "their", "theirs",
    "this", "that", "these", "those",
    # Reflexive pronouns (these ARE concepts — "yourself", "myself" —
    # but they shouldn't be subjects of is_a/causes/etc. either)
    "myself", "yourself", "himself", "herself", "itself",
    "ourselves", "yourselves", "themselves",
    # Conjunctions
    "and", "or", "but", "if", "unless", "because", "since", "while",
    "although", "though", "whereas", "whether", "so", "yet", "nor",
    # Determiners / articles
    "a", "an", "the", "some", "any", "all", "each", "every", "both",
    "either", "neither", "no", "another", "such", "other",
    # Auxiliary / modal verbs
    "is", "are", "was", "were", "be", "been", "being", "am",
    "do", "does", "did", "have", "has", "had", "having",
    "can", "could", "will", "would", "shall", "should", "may",
    "might", "must", "ought",
    # Prepositions (common ones)
    "in", "on", "at", "to", "for", "of", "with", "by", "from",
    "as", "into", "through", "during", "before", "after", "above",
    "below", "between", "under", "over", "against", "about",
    # Negation / quantifiers
    "not", "only", "also", "too", "very", "just",
    # Existential / locative adverbs (function words, not concepts)
    "there", "here",
    # Indefinite pronouns (never start a real concept)
    "something", "anything", "nothing", "everything",
    "someone", "anyone", "everyone", "nobody",
    "somebody", "anybody", "everybody",
})
_CODE_PREFIXES: tuple[str, ...] = (
    "python:", "rust:", "man:", "wikipedia:", "wordnet:",
)
_STRUCTURAL_PREFIXES: tuple[str, ...] = (
    "_cat:", "_col:", "identity:", "_utt:",
    # Task/competence machinery — internal markers, not world knowledge.
    "skill:", "goal:", "domain:", "spatial:", "var:", "type:",
)
def is_world_concept(concept_id: str) -> bool:
    """Return True if *concept_id* looks like genuine world knowledge.

    This filters out:
    - **Code symbols** (``python:mind.deny_site``, ``rust:state.neurochemical``)
      — valid for self-introspection but not world knowledge.
    - **Structural hubs** (``_cat:emotion:joy``, ``identity:genesis``) —
      internal scaffolding.
    - **Function words** (``and``, ``it``, ``is``) — grammatical, not
      conceptual.
    - **Conversation fragments** (``i know the computer``, ``are related
      because genesis``) — misparses, not concepts.
    - **Too-short names** (< 3 chars after stripping suffix/prefix).
    - **Names with digits** (dates, IDs, line numbers).

    The check is purely syntactic — it does not consult the network.
    Callers that need a stronger check can also verify the concept has
    a definition or sufficient edges.
    """
    if not concept_id or not concept_id.strip():
        return False

    cid = concept_id.strip()

    # Reject code-namespace concepts
    if cid.startswith(_CODE_PREFIXES):
        return False

    # Reject structural hubs
    if cid.startswith(_STRUCTURAL_PREFIXES):
        return False

    # Strip polysemy sense suffix for the remaining checks
    bare = strip_sense_suffix(cid).lower().strip()

    if not bare:
        return False

    # Reject function words (closed-class grammatical words)
    if bare in _FUNCTION_WORDS:
        return False

    # Reject names with digits (dates, line numbers, IDs)
    if any(ch.isdigit() for ch in bare):
        return False

    # Reject names that are too short (< 3 chars)
    if len(bare) < 3:
        return False

    # Reject conversation fragments — multi-word phrases that start
    # with a pronoun, auxiliary verb, or other function word. These
    # are misparses like "i know the computer" or "are related because
    # genesis". A real multi-word concept starts with a content word
    # (noun/adjective): "neural activity", "supreme court".
    words = bare.split()
    if len(words) > 1 and words[0] in _FUNCTION_WORDS:
        return False

    # Also reject underscore-separated conversation fragments.
    # Concept IDs use underscores as word separators, so phrases
    # like "i_don't_feel_great_about_this" (from vocabulary seeding)
    # are conversation fragments, not concepts. Real underscored
    # concepts like "neural_network" or "machine_learning" start
    # with content words, not function words.
    # Contractions are handled by checking the base form before
    # the apostrophe: "i'm" → "i", "there's" → "there".
    if "_" in bare:
        under_words = bare.split("_")
        if len(under_words) > 1:
            first = under_words[0].split("'")[0]
            if first in _FUNCTION_WORDS:
                return False

    # Reject verb-phrase fragments — multi-word phrases that contain
    # a verb inflection (past tense, -ing form) as a non-initial word.
    # These are extractor misparses like "antiquary tended" or
    # "neurons produced" where a past-tense verb was mistaken for
    # part of a noun phrase. Real noun phrases don't contain verb
    # inflections: "neural activity", "supreme court", "bill of rights".
    # We exclude "of" from this check because it legitimately appears
    # inside noun phrases ("bill of rights", "court of appeals").
    #
    # We only check against _VERB_INFLECTION_BREAKS — a conservative
    # subset of _BREAK_TOKENS that are clearly verb inflections and
    # NOT commonly used as adjectives. Past participles like "derived"
    # (brain-derived), "shown" (above-shown), "exhibited" are excluded
    # because they appear adjectivally in real concepts.
    if len(words) > 1:
        for w in words[1:]:
            if w == "of":
                continue
            if w in _VERB_INFLECTION_BREAKS:
                return False

    # Reject phrases that look like code paths even without a prefix
    # (e.g. "mind.deny_site", "concept_network.conceptnetwork")
    if "." in bare and "_" in bare:
        return False

    return True
_QUALITY_EDGE_SATURATION = 5
_QUALITY_TYPED_EDGE_SATURATION = 3
QUALITY_THRESHOLD = 0.15
def _is_verb_inflection(v: str, verb_set: frozenset[str]) -> bool:
    """Return True if v is clearly a verb inflection, not a noun base."""
    # Auxiliary / copula are always function-word-like.
    if v in ("is", "are", "was", "were", "am", "be", "been", "being",
             "have", "has", "had", "do", "does", "did", "done", "doing"):
        return True
    # If adding a common inflection yields another verb in the set, v is
    # likely a base form and may also be a noun; do not break on it.
    if v + "s" in verb_set or v + "es" in verb_set:
        return False
    if v.endswith("y") and (v[:-1] + "ies") in verb_set:
        return False
    if v.endswith("e") and (v[:-1] + "ing") in verb_set:
        return False
    if v + "ed" in verb_set or v + "ing" in verb_set:
        return False
    return True
_BREAK_TOKENS: frozenset[str] = _FUNCTION_WORDS | frozenset(
    v for v in _RELATION_VERBS if _is_verb_inflection(v, _RELATION_VERBS)
) | {
    # Phrase-breaking verb bases that the inflection heuristic above
    # skipped (because they have a plural / -s form in _RELATION_VERBS)
    # but which are overwhelmingly verbs in encyclopedic text.
    "include", "involve", "contain", "consist", "compose", "comprise",
    "constitute", "produce", "regulate", "control", "modulate", "mediate",
    "induce", "trigger", "generate", "create", "cause", "lead", "enable",
    "allow", "permit", "require", "depend", "originate", "arise",
    "inhibit", "suppress", "block", "prevent", "promote", "facilitate",
    "strengthen", "enhance", "denote", "describe", "define", "refer",
    "indicate", "signify", "mean", "express", "encode", "store", "retrieve",
    "compute", "calculate", "measure", "detect", "sense", "perceive",
    "recognize", "identify", "classify", "organize", "arrange", "structure",
    "operate", "act", "behave", "perform", "execute", "achieve",
    "accomplish", "complete", "finish", "occur", "happen", "appear",
    "emerge", "develop", "change", "vary", "differ", "correspond",
    "compare", "exceed", "dominate", "manage", "direct", "guide",
    "maintain", "communicate", "signal", "transmit", "propagate", "spread",
    "diffuse", "transfer", "transport", "release", "bind", "link", "join",
    "attach", "combine", "merge", "separate", "divide", "split", "share",
    "exchange", "replace", "respond", "react", "interact", "modify",
    "alter", "adapt", "adjust", "transform", "convert", "mutate", "evolve",
    "circulate", "flow", "cycle", "oscillate", "fluctuate", "migrate",
    "grow", "shrink", "expand", "contract", "extend", "open", "close",
    "break", "fall", "rise", "drop", "begin", "continue", "remain",
    "stay", "keep", "hold", "bring", "carry", "send", "receive", "obtain",
    "acquire", "gain", "lose", "pay", "buy", "sell", "spend", "meet",
    "leave", "enter", "reach", "approach", "pass", "cover", "fill",
    "build", "construct", "repair", "fix", "throw", "catch", "hit",
    "touch", "taste", "smell", "sound", "seem", "become", "try", "attempt",
    "fail", "succeed", "help", "assist", "aid", "serve", "work", "play",
    "rest", "wake", "feed", "eat", "drink", "breathe", "exhale", "inhale",
    "secrete", "absorb", "excrete", "digest", "metabolize", "synthesize",
    "degrade", "depolarize", "repolarize", "hyperpolarize",
    # Verbs added to _RELATION_VERBS that also need to be explicit
    # break tokens (they have -s forms so _is_verb_inflection skips them)
    "tend", "result", "derive", "show", "demonstrate", "reveal",
    "suggest", "yield", "exhibit", "display", "characterize",
}
_VERB_INFLECTION_BREAKS: frozenset[str] = frozenset({
    # -ed forms that are clearly verbal, not adjectival
    "tended", "resulted", "showed", "demonstrated", "revealed",
    "suggested", "yielded",
    # -ing forms (gerunds are nouns, but progressive forms are verbal)
    "causing", "leading", "emerging", "arising", "stemming",
    "originating", "depending", "requiring", "relying", "needing",
    "enabling", "supporting", "enhancing", "strengthening",
    "facilitating", "promoting", "preventing", "blocking",
    "suppressing", "inhibiting", "exciting", "activating",
    "deactivating", "stimulating", "creating", "forming",
    "generating", "producing", "triggering", "inducing",
    "provoking", "harming", "impairing", "damaging", "degrading",
    "disrupting", "resembling", "contradicting", "restoring",
    "regulating", "controlling", "modulating", "increasing",
    "decreasing", "reducing", "elevating", "lowering", "improving",
    "worsening", "affecting", "influencing", "determining",
    "mediating", "underlying", "subserving", "sustaining",
    "permitting", "allowing", "protecting", "shielding", "defending",
    "clearing", "removing", "rebuilding", "including", "involving",
    "containing", "consisting", "composing", "comprising",
    "constituting", "representing", "denoting", "describing",
    "defining", "referring", "indicating", "signifying", "meaning",
    "expressing", "encoding", "storing", "retrieving", "processing",
    "computing", "calculating", "measuring", "detecting", "sensing",
    "perceiving", "recognizing", "identifying", "classifying",
    "organizing", "arranging", "structuring", "functioning",
    "operating", "acting", "behaving", "performing", "executing",
    "achieving", "accomplishing", "completing", "finishing",
    "occurring", "happening", "appearing", "developing", "changing",
    "varying", "differing", "corresponding", "comparing", "exceeding",
    "dominating", "managing", "directing", "guiding", "maintaining",
    "communicating", "signaling", "transmitting", "propagating",
    "spreading", "diffusing", "transferring", "transporting",
    "releasing", "binding", "linking", "joining", "attaching",
    "combining", "merging", "separating", "dividing", "splitting",
    "sharing", "exchanging", "replacing", "responding", "reacting",
    "interacting", "modifying", "altering", "adapting", "adjusting",
    "transforming", "converting", "mutating", "evolving",
    "circulating", "flowing", "cycling", "oscillating",
    "fluctuating", "migrating", "growing", "shrinking", "expanding",
    "contracting", "extending", "opening", "closing", "breaking",
    "falling", "rising", "dropping", "beginning", "continuing",
    "remaining", "staying", "keeping", "holding", "bringing",
    "carrying", "sending", "receiving", "obtaining", "acquiring",
    "gaining", "losing", "paying", "buying", "selling", "spending",
    "meeting", "leaving", "entering", "reaching", "approaching",
    "passing", "covering", "filling", "building", "constructing",
    "repairing", "fixing", "throwing", "catching", "hitting",
    "touching", "tasting", "smelling", "sounding", "seeming",
    "becoming", "trying", "attempting", "failing", "succeeding",
    "helping", "assisting", "aiding", "serving", "working", "playing",
    "resting", "waking", "feeding", "eating", "drinking", "breathing",
    "exhaling", "inhaling", "secreting", "absorbing", "excreting",
    "digesting", "metabolizing", "synthesizing", "depolarizing",
    "repolarizing", "hyperpolarizing",
    # 3rd person singular forms (clearly verbal)
    "causes", "leads", "emerges", "arises", "stems", "originates",
    "depends", "requires", "relies", "needs", "enables", "supports",
    "enhances", "strengthens", "facilitates", "promotes", "prevents",
    "blocks", "suppresses", "inhibits", "excites", "activates",
    "deactivates", "stimulates", "creates", "forms", "generates",
    "produces", "triggers", "induces", "provokes", "harms", "impairs",
    "damages", "degrades", "disrupts", "resembles", "contradicts",
    "restores", "regulates", "controls", "modulates", "increases",
    "decreases", "reduces", "elevates", "lowers", "improves",
    "worsens", "affects", "influences", "determines", "mediates",
    "underlies", "sustains", "permits", "allows", "protects",
    "shields", "defends", "clears", "removes", "rebuilds", "includes",
    "involves", "contains", "consists", "composes", "comprises",
    "constitutes", "represents", "denotes", "describes", "defines",
    "refers", "indicates", "signifies", "means", "expresses",
    "encodes", "stores", "retrieves", "processes", "computes",
    "calculates", "measures", "detects", "senses", "perceives",
    "recognizes", "identifies", "classifies", "organizes", "arranges",
    "structures", "functions", "operates", "acts", "behaves",
    "performs", "executes", "achieves", "accomplishes", "completes",
    "finishes", "occurs", "happens", "appears", "develops", "changes",
    "varies", "differs", "corresponds", "compares", "exceeds",
    "dominates", "manages", "directs", "guides", "maintains",
    "communicates", "signals", "transmits", "propagates", "spreads",
    "diffuses", "transfers", "transports", "releases", "binds",
    "links", "joins", "attaches", "combines", "merges", "separates",
    "divides", "splits", "shares", "exchanges", "replaces",
    "responds", "reacts", "interacts", "modifies", "alters",
    "adapts", "adjusts", "transforms", "converts", "mutates",
    "evolves", "circulates", "flows", "cycles", "oscillates",
    "fluctuates", "migrates", "grows", "shrinks", "expands",
    "contracts", "extends", "opens", "closes", "breaks", "falls",
    "rises", "drops", "begins", "continues", "remains", "stays",
    "keeps", "holds", "brings", "carries", "sends", "receives",
    "obtains", "acquires", "gains", "loses", "pays", "buys", "sells",
    "spends", "meets", "leaves", "enters", "reaches", "approaches",
    "passes", "covers", "fills", "builds", "constructs", "repairs",
    "fixes", "throws", "catches", "hits", "touches", "tastes",
    "smells", "sounds", "seems", "becomes", "tries", "attempts",
    "fails", "succeeds", "helps", "assists", "aids", "serves",
    "works", "plays", "rests", "wakes", "feeds", "eats", "drinks",
    "breathes", "exhales", "inhales", "secretes", "absorbs",
    "excretes", "digests", "metabolizes", "synthesizes",
    "depolarizes", "repolarizes", "hyperpolarizes",
    # New verbs
    "tends", "results", "derives", "shows", "demonstrates",
    "reveals", "suggests", "yields", "exhibits", "displays",
    "characterizes",
})
def _normalize_id(name: str) -> str:
    """Normalize a concept name to a canonical form for ID matching.

    This is SYNTACTIC normalization only — it makes sure that
    "micro-animal", "Micro-Animal", "micro  animal", and
    "micro animal" all resolve to the same concept. It does NOT
    do semantic normalization (plural→singular, article stripping)
    — that's the caller's responsibility.

    Specifically:
    - Lowercase
    - Strip leading/trailing whitespace
    - Collapse internal whitespace to single spaces
    - Replace hyphens with spaces
    """
    return _WHITESPACE_RE.sub(" ", name.strip().lower().replace("-", " "))
def _extract_trigrams(text: str) -> frozenset[str]:
    """Extract all 3-character substrings from text.

    Used by the search index: a concept containing query Q as a
    substring must contain all trigrams of Q. The inverse doesn't
    hold (trigram presence ≠ substring presence), so the scoring
    step filters false positives.

    Returns an empty frozenset for strings shorter than 3 characters.
    """
    if len(text) < 3:
        return frozenset()
    return frozenset(text[i : i + 3] for i in range(len(text) - 2))
_LIVING_KEYWORDS: frozenset[str] = frozenset(
    {
        # common animals
        "dog",
        "cat",
        "horse",
        "bird",
        "fish",
        "snake",
        "cow",
        "pig",
        "sheep",
        "goat",
        "chicken",
        "duck",
        "rabbit",
        "mouse",
        "rat",
        "lion",
        "tiger",
        "bear",
        "wolf",
        "fox",
        "deer",
        "elephant",
        "monkey",
        "ape",
        "whale",
        "dolphin",
        "shark",
        "eagle",
        "hawk",
        "owl",
        "sparrow",
        "robin",
        "bee",
        "ant",
        "spider",
        "butterfly",
        "worm",
        "frog",
        "turtle",
        "lizard",
        "crocodile",
        "insect",
        "animal",
        "mammal",
        "reptile",
        "amphibian",
        "predator",
        "prey",
        "creature",
        "beast",
        "pet",
        "livestock",
        "wildlife",
        # body parts
        "brain",
        "heart",
        "lung",
        "liver",
        "kidney",
        "stomach",
        "blood",
        "bone",
        "skull",
        "spine",
        "rib",
        "muscle",
        "nerve",
        "neuron",
        "skin",
        "hair",
        "eye",
        "ear",
        "nose",
        "mouth",
        "tongue",
        "tooth",
        "hand",
        "finger",
        "arm",
        "leg",
        "foot",
        "knee",
        "elbow",
        "shoulder",
        "neck",
        "head",
        "face",
        "chest",
        "back",
        "belly",
        "organ",
        "tissue",
        "cell",
        "artery",
        "vein",
        # plants
        "tree",
        "flower",
        "grass",
        "leaf",
        "root",
        "stem",
        "branch",
        "seed",
        "fruit",
        "bark",
        "moss",
        "fern",
        "shrub",
        "bush",
        "vine",
        "algae",
        "fungus",
        "mushroom",
        "plant",
        "vegetation",
        "forest",
        "garden",
        "crop",
        "grain",
        "wheat",
        "corn",
        "rice",
        "oak",
        "pine",
        "maple",
        "birch",
        "rose",
        "lily",
        "tulip",
        # biological terms
        "organism",
        "species",
        "gene",
        "dna",
        "protein",
        "enzyme",
        "hormone",
        "metabolism",
        "digestion",
        "respiration",
        "reproduction",
        "evolution",
        "ecosystem",
        "habitat",
        "parasite",
        "bacteria",
        "virus",
        "infection",
        "immune",
        "antibody",
        "biological",
        "living",
        "alive",
        "animate",
        "life",
        "birth",
        "death",
        "growth",
        "aging",
        "maturation",
    }
)
_NON_LIVING_KEYWORDS: frozenset[str] = frozenset(
    {
        # tools
        "hammer",
        "screwdriver",
        "wrench",
        "pliers",
        "saw",
        "drill",
        "chisel",
        "file",
        "plane",
        "level",
        "measure",
        "ruler",
        "compass",
        "knife",
        "axe",
        "shovel",
        "rake",
        "hoe",
        "scythe",
        "needle",
        "scissors",
        "tongs",
        "clamp",
        "vise",
        "anvil",
        "tool",
        "implement",
        "instrument",
        "utensil",
        "device",
        # mechanical / machines
        "engine",
        "motor",
        "gear",
        "lever",
        "pulley",
        "wheel",
        "axle",
        "bearing",
        "spring",
        "valve",
        "pump",
        "turbine",
        "generator",
        "battery",
        "circuit",
        "wire",
        "cable",
        "switch",
        "machine",
        "mechanism",
        "apparatus",
        "contraption",
        "robot",
        "computer",
        "processor",
        "chip",
        "transistor",
        "sensor",
        "vehicle",
        "car",
        "truck",
        "bicycle",
        "train",
        "airplane",
        "ship",
        "boat",
        "rocket",
        "satellite",
        # materials
        "metal",
        "steel",
        "iron",
        "copper",
        "aluminum",
        "gold",
        "silver",
        "wood",
        "plastic",
        "rubber",
        "glass",
        "ceramic",
        "concrete",
        "brick",
        "stone",
        "sand",
        "clay",
        "fabric",
        "cotton",
        "wool",
        "silk",
        "leather",
        "paper",
        "cardboard",
        "material",
        "substance",
        "alloy",
        "composite",
        "fiber",
        # buildings / structures
        "house",
        "building",
        "tower",
        "bridge",
        "wall",
        "roof",
        "floor",
        "ceiling",
        "door",
        "window",
        "stair",
        "elevator",
        "factory",
        "warehouse",
        "barn",
        "garage",
        "shed",
        "fence",
        "road",
        "path",
        "tunnel",
        "dam",
        "canal",
        "pier",
        "dock",
        "structure",
        "construction",
        "architecture",
        "foundation",
    }
)
_ABSTRACT_KEYWORDS: frozenset[str] = frozenset(
    {
        # emotions / feelings
        "joy",
        "happiness",
        "sadness",
        "anger",
        "fear",
        "disgust",
        "surprise",
        "love",
        "hate",
        "hope",
        "despair",
        "pride",
        "shame",
        "guilt",
        "envy",
        "jealousy",
        "gratitude",
        "compassion",
        "empathy",
        "sympathy",
        "anxiety",
        "excitement",
        "boredom",
        "contentment",
        "frustration",
        "resentment",
        "nostalgia",
        "emotion",
        "feeling",
        "mood",
        "sentiment",
        "passion",
        "desire",
        "ambition",
        "motivation",
        "attitude",
        # relationships / social
        "friendship",
        "marriage",
        "family",
        "partnership",
        "alliance",
        "rivalry",
        "cooperation",
        "competition",
        "trust",
        "betrayal",
        "loyalty",
        "respect",
        "honor",
        "duty",
        "responsibility",
        "obligation",
        "commitment",
        "promise",
        "contract",
        "agreement",
        "relationship",
        "connection",
        "bond",
        "kinship",
        "community",
        "society",
        "culture",
        "tradition",
        "custom",
        "norm",
        # philosophical / conceptual
        "truth",
        "falsehood",
        "beauty",
        "justice",
        "freedom",
        "liberty",
        "equality",
        "democracy",
        "power",
        "authority",
        "wisdom",
        "knowledge",
        "ignorance",
        "belief",
        "doubt",
        "faith",
        "reason",
        "logic",
        "rationality",
        "intuition",
        "cognition",
        "awareness",
        "perception",
        "reality",
        "existence",
        "essence",
        "nature",
        "identity",
        "self",
        "mind",
        "soul",
        "spirit",
        "virtue",
        "morality",
        "ethics",
        "principle",
        "value",
        "meaning",
        "purpose",
        "destiny",
        "philosophy",
        "metaphysics",
        "epistemology",
        "ontology",
        # mathematical / formal
        "number",
        "equation",
        "function",
        "variable",
        "constant",
        "theorem",
        "proof",
        "axiom",
        "lemma",
        "corollary",
        "algebra",
        "geometry",
        "calculus",
        "statistics",
        "probability",
        "matrix",
        "vector",
        "tensor",
        "scalar",
        "dimension",
        "infinity",
        "limit",
        "derivative",
        "integral",
        "sum",
        "mathematics",
        "set",
        "graph",
        "topology",
        # general abstract
        "idea",
        "concept",
        "thought",
        "theory",
        "hypothesis",
        "rule",
        "law",
        "method",
        "strategy",
        "plan",
        "goal",
        "objective",
        "ideal",
        "standard",
        "quality",
        "quantity",
        "property",
        "attribute",
        "similarity",
        "difference",
        "pattern",
        "system",
        "process",
        "framework",
        "model",
    }
)
_EVENT_SUFFIXES: tuple[str, ...] = ("tion", "ment", "sion", "ance", "ence")
_NON_EVENT_STOPWORDS: frozenset[str] = frozenset(
    {
        # -tion: objects, places, abstract concepts (not processes)
        "nation",
        "station",
        "portion",
        "section",
        "fraction",
        "fiction",
        "option",
        "position",
        "institution",
        "constitution",
        "direction",
        "question",
        "notion",
        "potion",
        "lotion",
        "condition",
        "edition",
        "vacation",
        "location",
        "occupation",
        "population",
        "attention",
        "intention",
        "devotion",
        "addition",
        # -ment: objects, substances, abstract concepts
        "moment",
        "element",
        "fragment",
        "pigment",
        "garment",
        "ornament",
        "document",
        "cement",
        "basement",
        "apartment",
        "environment",
        "equipment",
        "regiment",
        "pavement",
        # -sion: objects, abstract concepts
        "vision",
        "version",
        "television",
        "lesion",
        # -ance: objects, abstract concepts
        "distance",
        "instance",
        "balance",
        "finance",
        "glance",
        "entrance",
        "stance",
        "chance",
        "abundance",
        # -ence: objects, abstract concepts
        "evidence",
        "sense",
        "silence",
        "absence",
        "presence",
        "patience",
        "sentence",
        "innocence",
        "expense",
        "reference",
        "preference",
        "influence",
        "sequence",
    }
)
_EVENT_KEYWORDS: frozenset[str] = frozenset(
    {
        "action",
        "event",
        "occurrence",
        "happening",
        "reaction",
        "response",
        "movement",
        "change",
        "transformation",
        "creation",
        "destruction",
        "decline",
        "collapse",
        "emergence",
        "development",
        "revolution",
        "operation",
        "execution",
        "performance",
        "activity",
        "behavior",
        "interaction",
        "communication",
        "negotiation",
        "conflict",
        "war",
        "battle",
        "fight",
        "race",
        "journey",
        "voyage",
        "expedition",
        "mission",
        "campaign",
        "project",
        "task",
        "job",
        "work",
        "labor",
        "effort",
        "attempt",
        "experiment",
        "test",
        "trial",
        "search",
        "discovery",
        "invention",
        "innovation",
        "breakthrough",
        "progress",
        "learning",
        "teaching",
        "training",
        "practice",
        "rehearsal",
        "celebration",
        "ceremony",
        "ritual",
        "festival",
        "party",
        "meeting",
        "gathering",
        "assembly",
        "conference",
        "summit",
    }
)
def detect_category(concept_name: str, definition: str = "") -> ConceptCategory:
    """Detect the category of a concept from its name and optional definition.

    Uses keyword matching to categorize concepts into living, non-living,
    abstract, or event categories. This models the biological distinction
    between how the brain processes different categories of things:
    living things in the temporal subsystem, tools in frontoparietal regions,
    abstract concepts in a distributed prefrontal network.

    The detection is heuristic — it catches the common case. Concepts
    that don't match any category return UNKNOWN, and can be categorized
    later by more sophisticated methods (e.g., categorize_all()).

    Args:
        concept_name: The canonical name of the concept.
        definition: Optional definition text for additional context.

    Returns:
        The detected ConceptCategory, or UNKNOWN if no match.
    """
    # Normalize: lowercase, strip prefixes (python:, rust:, identity:),
    # take the last component after dots for code-style concept IDs.
    name = concept_name
    for prefix in ("python:", "rust:", "identity:", "code:"):
        if name.startswith(prefix):
            name = name[len(prefix) :]
            break
    if "." in name:
        name = name.split(".")[-1]
    # Split camelCase / snake_case to get the base word(s)
    parts = _CAMEL_SPLIT_RE.split(name)
    name_lower = name.lower()
    parts_lower = [p.lower() for p in parts if p]

    # Combine name and definition for keyword matching.
    # We split into individual words to avoid false substring matches
    # (e.g., "cat" matching inside "category", "dog" in "dogma").
    search_words = set(parts_lower)
    if definition:
        search_words.update(w.lower() for w in _WORD_RE.findall(definition))

    # ── Check living things ───────────────────────────────────
    # Match if the concept name (or any of its camelCase parts) is a
    # known living keyword, or if the definition contains living keywords.
    if search_words & _LIVING_KEYWORDS:
        return ConceptCategory.LIVING

    # ── Check non-living things ───────────────────────────────
    if search_words & _NON_LIVING_KEYWORDS:
        return ConceptCategory.NON_LIVING

    # ── Check abstract concepts ───────────────────────────────
    if search_words & _ABSTRACT_KEYWORDS:
        return ConceptCategory.ABSTRACT

    # ── Check events ──────────────────────────────────────────
    # First check explicit event keywords
    if search_words & _EVENT_KEYWORDS:
        return ConceptCategory.EVENT
    # Then check suffix patterns (tion, ment, sion, ance, ence).
    # Skip known non-event nouns that happen to end in these suffixes
    # (nation, moment, vision, distance, evidence...). Only apply to
    # single-word concepts — multi-word concepts are less reliably events.
    if (
        " " not in name_lower
        and len(name_lower) > 4
        and name_lower not in _NON_EVENT_STOPWORDS
    ):
        for suffix in _EVENT_SUFFIXES:
            if name_lower.endswith(suffix):
                return ConceptCategory.EVENT

    return ConceptCategory.UNKNOWN
_VISUAL_KEYWORDS: frozenset[str] = frozenset(
    {
        # colors
        "red",
        "blue",
        "green",
        "yellow",
        "orange",
        "purple",
        "pink",
        "black",
        "white",
        "gray",
        "brown",
        "color",
        "bright",
        "dark",
        # visual properties
        "shape",
        "round",
        "square",
        "flat",
        "tall",
        "wide",
        "narrow",
        "large",
        "small",
        "big",
        "tiny",
        "huge",
        "visible",
        "image",
        "picture",
        "pattern",
        "line",
        "curve",
        "angle",
        "surface",
        # visual objects
        "tree",
        "flower",
        "mountain",
        "river",
        "sky",
        "cloud",
        "star",
        "sun",
        "moon",
        "rainbow",
        "shadow",
        "light",
        "landscape",
        "scene",
        "view",
        "sight",
        "appearance",
        "look",
    }
)
_MOTOR_KEYWORDS: frozenset[str] = frozenset(
    {
        # verbs of motion
        "run",
        "walk",
        "jump",
        "swim",
        "climb",
        "crawl",
        "fly",
        "dive",
        "throw",
        "catch",
        "kick",
        "punch",
        "hit",
        "push",
        "pull",
        "lift",
        "carry",
        "grab",
        "grasp",
        "reach",
        "move",
        "dance",
        # body actions
        "breathe",
        "eat",
        "drink",
        "bite",
        "chew",
        "swallow",
        "blink",
        "nod",
        "shake",
        "wave",
        "point",
        "touch",
        "feel",
        "step",
        # tools / manipulable objects
        "hammer",
        "screwdriver",
        "wrench",
        "knife",
        "axe",
        "shovel",
        "rake",
        "needle",
        "scissors",
        "tool",
        "instrument",
        # action nouns
        "movement",
        "motion",
        "action",
        "gesture",
        "exercise",
        "sport",
        "game",
        "race",
        "leap",
        "sprint",
        "march",
    }
)
_AUDITORY_KEYWORDS: frozenset[str] = frozenset(
    {
        # sounds
        "sound",
        "noise",
        "music",
        "song",
        "melody",
        "rhythm",
        "beat",
        "tone",
        "pitch",
        "chord",
        "harmony",
        "tune",
        "note",
        "chime",
        "ring",
        "buzz",
        "hum",
        "whisper",
        "shout",
        "scream",
        "cry",
        "roar",
        "growl",
        "bark",
        "meow",
        "chirp",
        "tweet",
        "howl",
        "echo",
        "voice",
        "speech",
        "word",
        "syllable",
        "vowel",
        "consonant",
        "accent",
        "language",
        "dialogue",
        "conversation",
        "thunder",
        "explosion",
        "crash",
        "bang",
        "pop",
        "snap",
        "click",
        "clap",
        "drum",
        "guitar",
        "piano",
        "violin",
        "flute",
    }
)
_VERBAL_KEYWORDS: frozenset[str] = frozenset(
    {
        # linguistic / propositional
        "truth",
        "falsehood",
        "lie",
        "fact",
        "claim",
        "statement",
        "argument",
        "premise",
        "conclusion",
        "definition",
        "axiom",
        "theorem",
        "proof",
        "hypothesis",
        "theory",
        "law",
        "rule",
        "principle",
        "proposition",
        "sentence",
        "paragraph",
        "text",
        "document",
        "contract",
        "agreement",
        "promise",
        "oath",
        "vow",
        "pledge",
        "declaration",
        "decree",
        "edict",
        "verdict",
        "judgment",
        "ruling",
        "opinion",
        "belief",
        "doctrine",
        "dogma",
        "creed",
        "tenet",
        "maxim",
        "adage",
        "proverb",
        "saying",
        "quote",
        "quotation",
        "citation",
        "reference",
    }
)
_ABSTRACT_MODALITY_KEYWORDS: frozenset[str] = frozenset(
    {
        # emotions / feelings
        "joy",
        "sadness",
        "anger",
        "fear",
        "love",
        "hate",
        "hope",
        "despair",
        "pride",
        "shame",
        "guilt",
        "envy",
        "compassion",
        "emotion",
        "feeling",
        "mood",
        "sentiment",
        # philosophical
        "cognition",
        "awareness",
        "existence",
        "essence",
        "freedom",
        "justice",
        "beauty",
        "virtue",
        "morality",
        "ethics",
        "wisdom",
        "knowledge",
        "ignorance",
        "doubt",
        "faith",
        "reason",
        "rationality",
        "intuition",
        "meaning",
        "purpose",
        "destiny",
        "soul",
        "spirit",
        "infinity",
        # mathematical / formal
        "number",
        "equation",
        "function",
        "variable",
        "limit",
        "derivative",
        "integral",
        "matrix",
        "vector",
        "probability",
        "statistics",
        "algebra",
        "geometry",
    }
)
def detect_modality(concept_name: str, definition: str = "") -> ConceptModality:
    """Detect the sensorimotor modality of a concept.

    Uses keyword matching to classify concepts by their primary
    sensory-motor grounding. This models the embodied cognition
    hypothesis — concepts are grounded in the same sensorimotor
    systems that perceive and act on the world.

    The detection is heuristic. Concepts that don't match return
    UNKNOWN and can be classified later. If a concept matches multiple
    modalities, MIXED is returned.

    Args:
        concept_name: The canonical name of the concept.
        definition: Optional definition text for additional context.

    Returns:
        The detected ConceptModality, or UNKNOWN if no match.
    """
    # Normalize: lowercase, strip prefixes, take last component
    name = concept_name
    for prefix in ("python:", "rust:", "identity:", "code:"):
        if name.startswith(prefix):
            name = name[len(prefix) :]
            break
    if "." in name:
        name = name.split(".")[-1]
    parts = _CAMEL_SPLIT_RE.split(name)
    parts_lower = [p.lower() for p in parts if p]

    search_words = set(parts_lower)
    if definition:
        search_words.update(w.lower() for w in _WORD_RE.findall(definition))

    matches: list[ConceptModality] = []
    if search_words & _VISUAL_KEYWORDS:
        matches.append(ConceptModality.VISUAL)
    if search_words & _MOTOR_KEYWORDS:
        matches.append(ConceptModality.MOTOR)
    if search_words & _AUDITORY_KEYWORDS:
        matches.append(ConceptModality.AUDITORY)
    if search_words & _VERBAL_KEYWORDS:
        matches.append(ConceptModality.VERBAL)
    if search_words & _ABSTRACT_MODALITY_KEYWORDS:
        matches.append(ConceptModality.ABSTRACT)

    if len(matches) == 0:
        return ConceptModality.UNKNOWN
    if len(matches) == 1:
        return matches[0]
    return ConceptModality.MIXED
_PROTECTED_ORIGINS = frozenset({
    "stated", "learned", "taught", "cognition_lesson",
    "identity", "introspection", "wordnet", "vocabulary",
    "dictionary", "curriculum",
})
_BRIDGE_ORIGINS = frozenset({
    "semantic_bridge", "hub_attachment", "associative_bridge", "bridge",
    "discovered", "dream-validated", "analogy",
})
_SENTINEL_EDGE: Edge = Edge(source="", target="", relation=RelationType.RELATED_TO)
