"""English morphosyntax — shared inflectional engine.

Both sides of the language system need the same grammatical
competence: comprehension *deconjugates* verbs to find the
predicate ("runs" → "run"), while production *conjugates* them to
agree with the subject ("they run" vs "it runs"). Keeping that
knowledge in one module guarantees the two systems never disagree
about English.

What lives here:

- **Verb deconjugation** — inflected form → candidate base forms
  ("tries" → "try", "stopped" → "stop", "ran" → "run")
- **Verb conjugation** — base form → third-person singular or
  plural agreement ("run" → "runs", "be" → "is"/"are")
- **Copula selection** — "is" vs "are" from subject plurality
- **Indefinite article** — "a" vs "an" by *sound*, not letter
  ("an hour", "a university", "an MRI")
- **Plural noun-phrase detection** — coordinated NPs
  ("salt and pepper") and plural head nouns
- **Person pronoun selection** — she/he/it/they from the concept
  network's animacy category, gender property, and plurality

These are grammatical building blocks — the machinery of English,
not the content of what Genesis says. A relation verb seed is a
building block; this module is the inflectional system that makes
those blocks agree with their subjects.

References:
- Quirk, R., Greenbaum, S., Leech, G., & Svartvik, J. (1985).
  *A Comprehensive Grammar of the English Language*. Longman.
"""

from __future__ import annotations

import logging

__all__ = [
    "VERB_LEXICON",
    "agree_verb_phrase",
    "conjugate_verb",
    "copula",
    "deconjugate_verb",
    "indefinite_article",
    "is_plural_np",
    "is_verb_form",
    "person_pronoun",
]

logger = logging.getLogger(__name__)

# ─── Verb lexicon ───────────────────────────────────────────────
#
# Base forms of verbs Genesis knows. Used by comprehension to decide
# whether an inflected word is a verb ("contains" → "contain" ✓,
# "dogs" → "dog" ✗) and by production to conjugate relation verbs.
# Conservative: false negatives (missing a verb) are safer than false
# positives (calling a noun a verb).

_VERB_BASES: frozenset[str] = frozenset(
    {
        # Copula and auxiliaries
        "be", "am", "is", "are", "was", "were", "been", "being",
        "do", "does", "did", "done", "have", "has", "had",
        "will", "would", "shall", "should", "can", "could",
        "may", "might", "must", "ought",
        # Relation verbs (the concept network's typed relations)
        "cause", "enable", "lead", "create", "produce", "harm",
        "prevent", "block", "stop", "contradict", "depend",
        "require", "need", "emerge", "arise", "relate", "connect",
        "resemble", "contain", "include", "comprise", "consist",
        "aim", "precede", "follow", "act", "affect", "characterize",
        "define", "express", "allow", "permit", "support", "enhance",
        "strengthen", "weaken", "hurt", "damage", "destroy", "build",
        "generate", "form", "constitute", "involve", "entail",
        "imply", "indicate", "suggest", "signal", "represent",
        "contribute", "give", "take", "stem", "derive", "result",
        "belong", "focus", "rely", "respond", "amount", "insist",
        "differ", "suffer", "participate", "account", "wonder",
        # Common action verbs
        "say", "tell", "know", "see", "think", "feel", "go", "come",
        "make", "find", "look", "seem", "show", "want", "use", "try",
        "help", "turn", "start", "live", "play", "run", "move",
        "believe", "hold", "bring", "happen", "write", "provide",
        "sit", "stand", "lose", "pay", "meet", "learn", "change",
        "understand", "watch", "win", "offer", "remember", "love",
        "consider", "appear", "buy", "wait", "serve", "die", "send",
        "expect", "stay", "fall", "cut", "reach", "remain", "raise",
        "pass", "sell", "report", "decide", "pull", "explain", "hope",
        "develop", "carry", "break", "receive", "agree", "hit",
        "eat", "cover", "catch", "draw", "choose", "walk", "drink",
        "sleep", "dream", "wake", "speak", "talk", "listen", "hear",
        "read", "ask", "answer", "call", "open", "close", "begin",
        "end", "finish", "continue", "keep", "let", "put", "set",
        "get", "become", "leave", "return", "arrive", "depart",
        "enter", "exit", "grow", "rise", "drive", "fly", "forget",
        "wear", "tear", "hide", "bear", "mean", "matter", "mind",
        "notice", "realize", "recognize", "remind",
        "imagine", "suppose", "guess", "wish", "prefer", "enjoy",
        "like", "dislike", "hate", "fear", "worry", "doubt", "trust",
        "accept", "reject", "deny", "admit", "claim", "state",
        "describe", "mention", "repeat", "add", "remove", "update",
        "fix", "check", "test", "verify", "compute", "calculate",
        "analyze", "compare", "sort", "count", "search", "process",
        "store", "load", "save", "print", "display", "render",
        "compose", "combine", "join", "split", "merge", "divide",
        "increase", "decrease", "reduce", "improve", "worsen",
        "protect", "attack", "defend", "fight", "exist", "occur",
        "persist", "vanish", "disappear", "last",
        "cost", "owe", "own", "possess", "lack", "gain", "earn",
        "spend", "waste", "share", "trade", "exchange",
        "work", "rest", "study", "practice", "train",
        "teach", "observe", "detect",
        "measure", "record", "collect", "gather", "assemble",
        "release", "absorb", "emit", "reflect", "transmit",
        "flow", "float", "sink", "freeze", "melt", "boil", "burn",
        "cool", "heat", "expand", "contract", "dissolve", "mix",
        "evaporate", "condense", "conduct", "insulate", "charge",
        # Additional common verbs
        "sing", "dance", "swim", "jump", "climb", "throw", "kick",
        "push", "lift", "drop", "shake", "stir", "pour",
        "wash", "clean", "cook", "bake", "plant", "water", "feed",
        "breathe", "blink", "smile", "laugh", "cry", "shout",
        "whisper", "scream", "whistle", "hum", "relax",
        "hurry", "rush", "chase", "guide",
        "seek", "hunt", "fish", "farm", "harvest",
        "thank", "greet", "welcome", "invite", "visit", "attend",
        "marry", "birth", "name", "touch", "grab",
        "kiss", "hug", "pat", "pet", "bite", "chew",
        "swallow", "spit", "cough", "sneeze", "yawn", "nap",
        "paint", "sculpt", "carve", "weave", "sew", "knit",
        "repair", "bend", "fold", "roll",
    }
)

# Irregular inflected → base mappings. Covers past tense, past
# participles, and the suppletive copula/auxiliary forms.
_IRREGULAR_BASE: dict[str, str] = {
    # Copula / auxiliaries
    "am": "be", "is": "be", "are": "be", "was": "be", "were": "be",
    "been": "be", "being": "be", "'m": "be", "'re": "be", "'s": "be",
    "does": "do", "did": "do", "done": "do",
    "has": "have", "had": "have",
    # Strong verbs — past tense
    "ran": "run", "went": "go", "came": "come", "saw": "see",
    "gave": "give", "took": "take", "made": "make", "got": "get",
    "wrote": "write", "spoke": "speak", "knew": "know",
    "thought": "think", "felt": "feel", "found": "find",
    "told": "tell", "said": "say", "brought": "bring",
    "bought": "buy", "built": "build", "sent": "send",
    "heard": "hear", "sat": "sit", "stood": "stand",
    "fell": "fall", "grew": "grow", "became": "become",
    "began": "begin", "broke": "break", "kept": "keep",
    "lost": "lose", "won": "win", "met": "meet", "left": "leave",
    "meant": "mean", "held": "hold", "slept": "sleep",
    "ate": "eat", "drank": "drink", "drove": "drive",
    "flew": "fly", "forgot": "forget", "chose": "choose",
    "froze": "freeze", "stole": "steal", "swam": "swim",
    "sang": "sing", "rang": "ring", "sank": "sink",
    "shook": "shake", "understood": "understand",
    "withstood": "withstand", "arose": "arise", "awoke": "awake",
    "bore": "bear", "wore": "wear", "tore": "tear",
    "hid": "hide", "lay": "lie", "laid": "lay", "led": "lead",
    "paid": "pay", "sold": "sell", "taught": "teach",
    "caught": "catch", "fought": "fight", "sought": "seek",
    "shot": "shoot", "spent": "spend", "split": "split",
    "spread": "spread", "struck": "strike", "stuck": "stick",
    "swung": "swing", "wept": "weep", "withdrew": "withdraw",
    # Past participles that differ from simple past
    "gone": "go", "seen": "see", "given": "give", "taken": "take",
    "known": "know", "written": "write", "spoken": "speak",
    "broken": "break", "fallen": "fall", "grown": "grow",
    "born": "bear", "borne": "bear", "worn": "wear",
    "torn": "tear", "hidden": "hide", "risen": "rise",
    "driven": "drive", "eaten": "eat", "drunk": "drink",
    "forgotten": "forget", "chosen": "choose", "frozen": "freeze",
    "stolen": "steal", "swum": "swim", "sung": "sing",
    "rung": "ring", "sunk": "sink", "shaken": "shake",
    "arisen": "arise", "lain": "lie",
}

# Singular nouns ending in 's' — these look plural but take "is".
# Used by both copula() and is_plural_np().
_SINGULAR_S_WORDS: frozenset[str] = frozenset(
    {
        "cognition", "awareness", "happiness", "sadness",
        "madness", "illness", "weakness", "darkness", "brightness",
        "fitness", "business", "laziness", "loneliness", "kindness",
        "goodness", "greatness", "openness", "stillness", "usefulness",
        "news", "mathematics", "physics", "ethics", "politics",
        "economics", "linguistics", "genetics", "logistics",
        "series", "species", "crisis", "analysis", "thesis",
        "hypothesis", "synthesis", "emphasis", "diagnosis",
        "genesis", "status", "virus", "lotus", "bus", "plus",
        "glass", "class", "grass", "mass", "pass", "boss",
        "loss", "cross", "dress", "press", "address", "process",
        "progress", "success", "access", "across", "alas",
        "chaos", "pathos", "cosmos", "bias", "canvas", "octopus",
        "hippocampus", "stimulus", "syllabus", "is", "was",
        "has", "does", "this", "thus", "yes", "his", "its",
        # 3sg verb forms rarely used as plural nouns — take singular
        # agreement. "makes" (3sg of "make") as a concept subject is
        # singular, not a plural noun. Consistent with "has"/"does"/"is"
        # already in this set.
        "makes",
    }
)


# ─── Deconjugation (analysis) ───────────────────────────────────


def deconjugate_verb(word: str) -> list[str]:
    """Return candidate base forms for an inflected verb.

    The candidates are ordered most-likely first. Callers check
    membership against ``VERB_LEXICON`` (or their own lexicon) —
    this function only performs the morphological stripping.

    Examples:
        "runs"    → ["run"]
        "tries"   → ["try", "trie"]
        "stopped" → ["stop", "stopp"]
        "making"  → ["make", "mak", "making"]
        "ran"     → ["run"]        (irregular)
        "is"      → ["be"]         (suppletive)
    """
    w = word.lower().strip().rstrip(",.!?;:")
    if not w:
        return []

    candidates: list[str] = [w]

    # Irregular forms first — they override suffix rules
    if w in _IRREGULAR_BASE:
        candidates.append(_IRREGULAR_BASE[w])

    # -ies → -y ("tries" → "try", "carries" → "carry")
    if len(w) > 3 and w.endswith("ies"):
        candidates.append(w[:-3] + "y")
        candidates.append(w[:-1])  # "tries" → "trie" (weaker)

    # -es for sibilant endings ("watches" → "watch", "goes" → "go")
    if len(w) > 2 and w.endswith(("shes", "ches", "sses", "xes", "zes", "oes")):
        candidates.append(w[:-2])

    # -s ("runs" → "run", "makes" → "make")
    if len(w) > 2 and w.endswith("s") and not w.endswith("ss"):
        candidates.append(w[:-1])

    # -ed ("walked" → "walk", "hoped" → "hope", "stopped" → "stop")
    if len(w) > 3 and w.endswith("ed"):
        candidates.append(w[:-2])            # walked → walk
        candidates.append(w[:-1])            # hoped → hope
        if len(w) > 4 and w[-3] == w[-4]:    # stopped → stopp → stop
            candidates.append(w[:-3])

    # -ing ("walking" → "walk", "making" → "make", "running" → "run")
    if len(w) > 4 and w.endswith("ing"):
        stem = w[:-3]
        candidates.append(stem + "e")        # making → make
        candidates.append(stem)              # walking → walk
        if len(stem) > 1 and stem[-1] == stem[-2]:
            candidates.append(stem[:-1])     # running → runn → run

    # Deduplicate while preserving order
    seen: set[str] = set()
    result: list[str] = []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            result.append(c)
    return result


def is_verb_form(
    word: str, lexicon: frozenset[str] | None = None
) -> str | None:
    """Return the base form if the word is a known verb, else None.

    Deconjugates the word and checks each candidate against the
    lexicon. This is what lets comprehension find "runs" (→ "run"),
    "thinks" (→ "think"), and "contains" (→ "contain") as verbs
    while rejecting plural nouns like "dogs" (→ "dog", not a verb).

    For irregular inflected forms (listed in ``_IRREGULAR_BASE``), the
    true base is preferred over the inflected word itself. Without
    this, ``is_verb_form("has")`` would return "has" (because "has" is
    in ``_VERB_BASES``) rather than "have" — the actual base form. That
    caused ``agree_verb_phrase`` to skip deconjugation and corrupt
    "has" → "ha" via naive 's'-stripping.
    """
    if lexicon is None:
        lexicon = _VERB_BASES
    w = word.lower().strip().rstrip(",.!?;:")
    # Irregular inflected forms: prefer the true base over the word
    # itself. "has" → "have", "is" → "be", "does" → "do".
    if w in _IRREGULAR_BASE:
        base = _IRREGULAR_BASE[w]
        if base in lexicon:
            return base
    for candidate in deconjugate_verb(word):
        if candidate in lexicon:
            return candidate
    return None


def is_participle(word: str, lexicon: frozenset[str] | None = None) -> bool:
    """True when the word is a past/passive participle of a verb.

    Covers regular "-ed"/"-en" forms ("needed", "eaten") and
    irregulars whose inflected form maps to a verb base ("sung" →
    "sing", "made" → "make", "written" → "write"). Used to detect
    passive constructions after copulas ("was sung by the choir").
    """
    if lexicon is None:
        lexicon = _VERB_BASES
    w = word.lower().strip().rstrip(",.!?;:")
    if not w:
        return False
    # Irregular inflected form with a verb base
    if w in _IRREGULAR_BASE and _IRREGULAR_BASE[w] in lexicon:
        return True
    # Regular participle: -ed / -en stripping reaches a verb base
    if w.endswith(("ed", "en")):
        base = is_verb_form(w, lexicon)
        return base is not None and base != w
    return False


VERB_LEXICON = _VERB_BASES


# ─── Conjugation (synthesis) ────────────────────────────────────


def conjugate_verb(base: str, plural: bool = False, past: bool = False) -> str:
    """Conjugate a base-form verb for present or past tense.

    Args:
        base: The uninflected verb ("run", "make", "be").
        plural: True when the subject takes plural agreement
            ("they run", "dogs run", "we run") — i.e. everything
            except third-person singular.
        past: True for simple past tense.

    Returns the inflected form. Unknown words fall back to the
    regular rules.
    """
    b = base.lower().strip()

    if past:
        # Irregular past — return the simple-past form (not a
        # participle like "gone"/"eaten")
        for inflected, stem in _IRREGULAR_BASE.items():
            if stem == b and inflected in _SIMPLE_PAST:
                return inflected
        # Regular: -ed
        if b.endswith("e"):
            return b + "d"
        if len(b) > 2 and b.endswith("y") and b[-2] not in "aeiou":
            return b[:-1] + "ied"
        if (len(b) <= 4 and len(b) > 2
                and b[-1] not in "aeiouwyx" and b[-2] in "aeiou"
                and b[-3] not in "aeiou"):
            return b + b[-1] + "ed"  # stop → stopped
        return b + "ed"

    if plural:
        return b

    # Third-person singular
    if b == "be":
        return "is"
    if b == "have":
        return "has"
    if b == "do":
        return "does"
    if b.endswith(("s", "x", "z", "ch", "sh", "o")):
        return b + "es"
    if len(b) > 1 and b.endswith("y") and b[-2] not in "aeiou":
        return b[:-1] + "ies"
    return b + "s"


# Simple-past forms (subset of irregulars — excludes participles).
_SIMPLE_PAST: frozenset[str] = frozenset(
    {
        "ran", "went", "came", "saw", "gave", "took", "made", "got",
        "wrote", "spoke", "knew", "thought", "felt", "found", "told",
        "said", "brought", "bought", "built", "sent", "heard", "sat",
        "stood", "fell", "grew", "became", "began", "broke", "kept",
        "lost", "won", "met", "left", "meant", "held", "slept",
        "ate", "drank", "drove", "flew", "forgot", "chose", "froze",
        "stole", "swam", "sang", "rang", "sank", "shook",
        "understood", "arose", "awoke", "bore", "wore", "tore",
        "hid", "lay", "laid", "led", "paid", "sold", "taught",
        "caught", "fought", "sought", "shot", "spent", "split",
        "spread", "struck", "stuck", "swung", "wept", "withdrew",
        "was", "were", "did", "had",
    }
)


def agree_verb_phrase(phrase: str, subject_is_plural: bool) -> str:
    """Adjust a verb phrase's first word for subject number.

    The verb seeds store third-person-singular forms ("enables",
    "is part of", "gives rise to"). When the subject is plural —
    "dreams", "they", "salt and pepper" — the first word must
    deconjugate: "enables" → "enable", "is" → "are".

    Examples:
        ("enables", True)  → "enable"
        ("is part of", True) → "are part of"
        ("gives rise to", True) → "give rise to"
        ("causes", False)  → "causes"   (unchanged)
    """
    if not subject_is_plural or not phrase:
        return phrase

    words = phrase.split(None, 1)
    first, rest = words[0], (words[1] if len(words) > 1 else "")

    # Copula and auxiliaries with suppletive plural forms.
    # "is" → "are", "has" → "have", "does" → "do".
    # These are handled explicitly because the generic deconjugation
    # path can corrupt them (e.g. naive 's'-stripping turns "has" into
    # "ha" rather than "have").
    if first == "is":
        return "are" + (" " + rest if rest else "")
    if first == "has":
        return "have" + (" " + rest if rest else "")
    if first == "does":
        return "do" + (" " + rest if rest else "")

    # Modals and auxiliaries that never inflect
    if first in ("can", "could", "will", "would", "shall", "should",
                 "may", "might", "must"):
        return phrase

    # Deconjugate the first word back to base form.
    # "enables" → "enable", "depends" → "depend", "does" → "do"
    base = is_verb_form(first)
    if base and base != first:
        # Re-attach the rest of the phrase
        return base + (" " + rest if rest else "")

    # Unknown verb — strip a bare 3sg 's' as a last resort
    if first.endswith("s") and not first.endswith(("ss", "us", "is")):
        return first[:-1] + (" " + rest if rest else "")

    return phrase


# ─── Copula ─────────────────────────────────────────────────────


def is_plural_np(text: str) -> bool:
    """Return True if a noun phrase is plural.

    Two signals:
    - Coordination: "salt and pepper", "mind and body"
    - Plural head noun: "dreams", "the cats"

    Conservative on s-final words: English has many singular
    nouns ending in 's' ("cognition", "physics", "genesis").
    """
    t = text.strip().lower()
    if not t:
        return False
    # Coordinated NPs are plural
    if " and " in t or " or " in t:
        return True
    # Check the last word (English head nouns are usually final)
    head = t.split()[-1].rstrip(",.!?;:")
    if head in _SINGULAR_S_WORDS:
        return False
    if head.endswith(("ss", "us", "is")):
        return False
    if head.endswith("s") and len(head) > 2:
        # -ies is plural only if base isn't in the singular set
        # ("bodies" plural vs "series" singular — series is caught
        # by _SINGULAR_S_WORDS above)
        return True
    return False


def copula(subject: str) -> str:
    """Pick "is" or "are" for a subject noun phrase."""
    return "are" if is_plural_np(subject) else "is"


# ─── Articles ───────────────────────────────────────────────────

# Words starting with a vowel *letter* but a consonant *sound*
# (/j/ "y-" or /w/ "w-") take "a", not "an".
_AN_EXCEPTIONS: frozenset[str] = frozenset(
    {
        "university", "unicorn", "unique", "unit", "united",
        "universe", "universal", "user", "use", "useful",
        "usual", "usually", "utopia", "utopian", "european",
        "eucalyptus", "euphemism", "one", "once", "one-time",
    }
)

# Words starting with a consonant letter but a vowel *sound*
# (silent h) take "an", not "a".
_A_EXCEPTIONS: frozenset[str] = frozenset(
    {"hour", "honest", "honor", "honour", "heir", "herb", "hourly"}
)


def indefinite_article(phrase: str) -> str:
    """Pick "a" or "an" for the following noun phrase, by sound.

    Examples:
        "apple" → "an", "banana" → "a", "hour" → "an",
        "university" → "a", "MRI" → "an" (letter name "em").
    """
    if not phrase:
        return "a"
    word = phrase.strip().split()[0].lower().rstrip(",.!?;:")
    if not word:
        return "a"

    if word in _A_EXCEPTIONS:
        return "an"
    if word in _AN_EXCEPTIONS:
        return "a"
    # Acronyms: all-caps letter names starting with a vowel-sound
    # letter (F→"eff", L→"el", M→"em", N→"en", R→"ar", S→"es",
    # X→"ex", A→"ay", E→"ee", I→"eye", O→"oh") take "an".
    raw = phrase.strip().split()[0]
    if len(raw) > 1 and raw.isupper():
        return "an" if raw[0] in "AEFHILMNORSX" else "a"
    return "an" if word[0] in "aeiou" else "a"


# ─── Pronouns ───────────────────────────────────────────────────

# Pronoun forms: (nominative, accusative, plural_agreement)
_PRONOUNS: dict[str, tuple[str, str, bool]] = {
    "she": ("she", "her", False),
    "he": ("he", "him", False),
    "they": ("they", "them", True),   # singular they → plural agreement
    "it": ("it", "it", False),
}


def person_pronoun(
    concept_id: str,
    network=None,
    display: str = "",
    for_object: bool = False,
) -> tuple[str, bool]:
    """Pick a pronoun for a concept from its animacy properties.

    Returns (pronoun, plural_agreement) where pronoun is the
    nominative or accusative form and plural_agreement tells the
    caller whether verbs should use plural inflection ("they run").

    Selection:
    - Genesis herself → "she" (her established identity)
    - Concept with gender property → "he"/"she"
    - Living person (category LIVING + proper noun or person type)
      → "they" (neutral when gender is unknown)
    - Plural noun phrase → "they"
    - Everything else → "it"

    The animacy information is *learned* — it comes from the
    concept's ``category`` (detected by ``detect_category``) and
    ``properties["gender"]``/``["proper_noun"]``, never a hardcoded
    name list.
    """
    cid = (concept_id or "").lower().strip()
    disp = (display or concept_id or "").lower().strip()

    # Genesis — established self-reference
    if cid == "genesis" or disp == "genesis":
        return _pick("she", for_object)

    # Plural / coordinated NPs
    if is_plural_np(disp or cid):
        return _pick("they", for_object)

    # Consult the concept network for animacy
    if network is not None and cid:
        concept = None
        try:
            concept = network.get_concept(concept_id)
            if concept is None:
                concept = network.get_concept(cid)
        except Exception:  # noqa: BLE001
            concept = None
        if concept is not None:
            gender = str(concept.properties.get("gender", "")).lower()
            if gender in ("female", "feminine", "woman", "girl"):
                return _pick("she", for_object)
            if gender in ("male", "masculine", "man", "boy"):
                return _pick("he", for_object)
            # Living things that are people (proper nouns or
            # explicitly person-typed) get "they" — gender-neutral
            # singular. Animals and plants stay "it".
            try:
                from ..concepts import ConceptCategory

                if concept.category == ConceptCategory.LIVING:
                    is_person = bool(concept.properties.get("proper_noun"))
                    if not is_person:
                        # is_a → person/human/user/creator?
                        for edge in network.get_edges(concept.id, "out"):
                            if edge.relation.value in ("is_a", "instance_of"):
                                t = edge.target.lower()
                                if t in ("person", "human", "user",
                                         "creator", "woman", "man",
                                         "girl", "boy", "people"):
                                    is_person = True
                                    break
                    if is_person:
                        return _pick("they", for_object)
            except Exception as e:  # noqa: BLE001
                logger.debug(f"animacy check failed for {concept_id}: {e}")

    return _pick("it", for_object)


def _pick(key: str, for_object: bool) -> tuple[str, bool]:
    nom, acc, plural = _PRONOUNS[key]
    return (acc if for_object else nom), plural
