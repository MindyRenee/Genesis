"""Problem intake — heard language → task spec, via comprehension.

The outer-world half of the puzzle loop. This module does not parse
English surface forms: it interprets the *semantic* output of the
``ComprehensionEngine`` — propositions (subject / predicate /
object / roles), negation, question focus, and speech act. Task
recognition is frame composition over that structure:

- a quantified containment proposition on an *indefinite* subject
  ("a tessel is a machine that HAS exactly two of: ...") is a rule;
- the same predicate on a *definite* subject ("alpha has ...") is an
  item description;
- a copula to a kind noun ("beta is not a wug") is a label
  assertion; negation carries the negative label;
- a wh-question over a plural kind ("which are tessels") is the
  membership query;
- an imperative placement verb whose object chunks into
  NP–relation–NP ("put the star left of the moon") is an
  arrangement;
- a continuation predicate over a mark sequence ("what comes next
  in a b a b") is a pattern-completion task;
- an achieve-verb with a numeric object and a counted instrument
  ("make 9 with groups of 2, 5 and 4") is a composition task.

Because the tests operate on semantic roles rather than word order,
rephrasings that preserve meaning ("a quark is any object containing
at least two of: ...") compile identically — surface variation is
comprehension's job. Anything the interpretation can't ground
honestly — an unstated rule with no answer key, a non-periodic
pattern, prose with no task frame — returns None and the input
continues down the normal conversational path. Returned specs are
*raw*; ``normalize_offered`` remains the validating gate.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..language.comprehension import ComprehensionEngine, ComprehensionResult

_WORD_NUM = {
    "a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4,
    "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
    "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
    "nineteen": 19, "twenty": 20,
}

# ── Verb senses (lexical semantics — predicate → role frame) ────
# Possession/containment: "X has Y" ascribes Y to X.
_CONTAINS = frozenset({
    "has", "have", "had", "contains", "contain", "includes",
    "include", "holds", "hold", "owns", "own", "got", "gets", "get",
    "sports", "sport", "carries", "carry", "bears", "bear",
    "needs", "need", "requires", "require",
    # Reduced relatives surface as gerund predicates ("an object
    # containing two parts" ≡ "that contains two parts").
    "containing", "having", "including", "holding", "needing",
    "requiring", "sporting", "bearing", "carrying",
})
# Placement: imperative verbs whose object is THEME + locations.
_PLACE = frozenset({
    "put", "place", "set", "lay", "position", "arrange", "stick",
    "move", "stand",
})
# Achieve-a-quantity: "make 9", "reach 12".
_ACHIEVE = frozenset({
    "make", "makes", "made", "making", "build", "builds", "built",
    "building", "reach", "reaches", "reached", "reaching", "total",
    "totals", "sum", "sums", "produce", "produces", "create",
    "creates", "created", "form", "forms", "combine", "combines",
})
# Continuation: the predicate asks what follows.
_PREDICT = frozenset({
    "comes", "come", "continues", "continue", "goes", "go",
    "follows", "follow", "repeats", "repeat", "extends", "extend",
    "finishes", "finish", "completes", "complete",
})
_COPULA = frozenset({"is", "are", "was", "were", "am", "be", "been", "being"})
# Locative verbs — the located thing is the subject, the relation
# phrase sits in the object ("the star goes left of the moon").
_LOCATIVE = frozenset({
    "goes", "go", "sits", "sit", "stands", "stand", "belongs",
    "belong", "stays", "stay", "lives", "live", "rests", "rest",
    "lies", "lie",
})

# ── NP machinery ────────────────────────────────────────────────
# Leading words that don't belong to an entity/attribute name.
_NP_LEAD = frozenset({
    "a", "an", "the", "some", "any", "each", "every", "all", "both",
    "no", "none", "only", "just", "exactly", "precisely", "of",
    "with", "having", "this", "that", "these", "those", "its",
    "another", "one", "ones", "my", "your", "not", "n't", "never",
})
# Category nouns — filler in enumerations ("these parts", "all three
# things"), never attribute names themselves.
_CATEGORY_NOUNS = frozenset({
    "part", "parts", "thing", "things", "item", "items", "piece",
    "pieces", "feature", "features", "property", "properties",
    "attribute", "attributes", "trait", "traits", "kind", "kinds",
    "type", "types", "of", "them", "following", "these", "those",
    "the", "one", "ones",
})
# Entities a proposition can name without naming a task thing —
# pronouns, fillers, discourse words.
_NON_ENTITY = frozenset({
    "i", "you", "he", "she", "it", "we", "they", "me", "him", "her",
    "us", "them", "itself", "something", "anything", "everything",
    "nothing", "one", "someone", "anyone", "everyone", "what",
    "which", "who", "this", "that", "these", "those", "there",
    "here", "now", "then", "so", "if",
})

_TOKEN_RE = re.compile(r"[a-z0-9][\w-]*")
_COORD_SPLIT_RE = re.compile(r",|\band\b|\bor\b")


def _num(tok: str) -> int | None:
    """Digits or a small number word → int, else None."""
    tok = tok.strip().lower()
    if tok.isdigit():
        return int(tok)
    return _WORD_NUM.get(tok)


def _tokens(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def _np_name(chunk: str) -> str:
    """NP chunk → entity/attribute name: determiners and quantifier
    words stripped, content joined ("the red star" → red_star)."""
    toks = [
        t for t in _tokens(chunk)
        if t not in _NP_LEAD and t not in _CATEGORY_NOUNS and _num(t) is None
    ]
    return "_".join(toks)


def _enumerate(text: str) -> list[str]:
    """A coordinated NP string → member names.

    "a lens, a spring, and a dial" → [lens, spring, dial]. This is
    semantic enumeration over a proposition field, not a sentence
    pattern — any surface order the parser delivers works.
    """
    names: list[str] = []
    for seg in _COORD_SPLIT_RE.split(text):
        name = _np_name(seg)
        if name:
            names.append(name)
    return names


def _is_indefinite(subject: str) -> bool:
    """True when the subject is a generic NP ("a tessel", "one",
    "machines") rather than a named individual ("alpha", "the moon").
    Generics carry rules; definites carry instances."""
    s = subject.strip().lower()
    if not s:
        return False
    first = s.split()[0]
    if first in ("a", "an", "any", "some", "each", "every"):
        return True
    # Generic pronouns read as kinds ("something has two parts").
    if s in ("something", "anything", "everything", "one"):
        return True
    if s in _NON_ENTITY:
        return False
    # "machines have gears" — a bare plural subject is generic.
    # Sentence-initial capitalization is not a proper-name signal.
    return len(s) > 1 and s.endswith("s")


def _deplural(word: str) -> str:
    """Naive singular — kind matching ("tessels" ↔ "tessel")."""
    w = word.strip().lower()
    if w.endswith("ies") and len(w) > 3:
        return w[:-3] + "y"
    if w.endswith("es") and len(w) > 3:
        return w[:-2]
    if w.endswith("s") and len(w) > 1:
        return w[:-1]
    return w


def _slug(text: str) -> str:
    words = re.findall(r"[a-z0-9]+", text.lower())
    return "_".join(words[:3]) or "problem"


def _name(family: str, spec: dict) -> str:
    """Deterministic offer name — re-describing the same problem
    re-lands on the same task; a different problem lands elsewhere."""
    digest = hashlib.sha1(
        json.dumps(spec, sort_keys=True).encode()
    ).hexdigest()[:6]
    return f"heard_{_slug(family)}_{digest}"


# ── Quantifier scope ────────────────────────────────────────────
# Compositional quantifier semantics over a containment object:
# [op-word(s)] [count] "of" <set>. Each op is a semantic operator on
# the set, not a sentence pattern.
_QUANT_LEAD: dict[str, str] = {
    "exactly": "exactly", "precisely": "exactly",
    "all": "all_of", "both": "all_of", "every": "all_of",
    "any": "any_of", "some": "any_of",
    "none": "none_of", "no": "none_of",
}


def _quantified(obj: str) -> tuple[str, int | None, str] | None:
    """Parse a quantified containment object → (op, count, set_rest).

    "exactly two of these parts: a, b, c" → (exactly, 2, "these
    parts: a, b, c"). Words keep their punctuation — the colon and
    commas are the enumeration's delimiters. A bare numeral only
    counts when it takes the partitive ("two of them"); "a crest"
    is the article "a", not a count. Returns None when the object
    has no quantifier structure.
    """
    words = obj.split()
    if not words:
        return None

    def w(k: int) -> str:
        return words[k].lower().rstrip(",.;:") if k < len(words) else ""

    i = 0
    op: str | None = None
    count: int | None = None
    # Multi-word quantifier openers, longest-match first.
    for phrase, qop in (
        ("no more than", "at_most"), ("at most", "at_most"),
        ("up to", "at_most"), ("at least", "at_least"),
        ("no fewer than", "at_least"), ("a minimum of", "at_least"),
        ("a maximum of", "at_most"), ("one or more", "at_least"),
    ):
        plen = len(phrase.split())
        if " ".join(w(k) for k in range(plen)) == phrase:
            op = qop
            i = plen
            break
    else:
        if w(0) in _QUANT_LEAD:
            op = _QUANT_LEAD[w(0)]
            i = 1
        elif w(0) in ("only", "just"):
            # "only two of" → exactly; bare "only a dial" stays a
            # plain enumeration (only marks exclusivity).
            n = _num(w(1))
            if n is not None and w(2) == "of":
                op, count, i = "exactly", n, 3
            elif n is not None:
                op, count, i = "exactly", n, 2
    # An explicit count needs the partitive ("two of them"), except
    # after a quantifier opener where it scopes the set directly.
    if count is None and i < len(words):
        n = _num(w(i))
        if n is not None:
            if w(i) == "one" and w(i + 1) == "of":
                op = op or "any_of"
            elif w(i + 1) == "of" or op is not None:
                count = n
                i += 1
    if i < len(words) and w(i) == "of":
        i += 1
    if op is None and count is None:
        return None
    if op is None:
        op = "exactly"  # bare partitive: "two of these parts"
    rest = " ".join(words[i:])
    return op, count, rest


def _predicate_from_object(obj: str, vocab: list[str]) -> dict | None:
    """Containment object → rule predicate.

    The set expression after the quantifier resolves three ways:
    an explicit enumeration after a colon ("these parts: a, b, c"),
    a direct enumeration ("of a crest and a tail"), or a bare
    category reference ("these parts") — which defers to the
    vocabulary of attributes the items themselves mention.
    """
    q = _quantified(obj)
    if q is not None:
        op, count, rest = q
        if ":" in rest:
            rest = rest.split(":", 1)[1]
        members = _enumerate(rest)
        if not members:
            # "these parts" / "the following" — vocabulary deferred.
            members = list(vocab)
        if not members:
            return None
        pred: dict[str, Any] = {"op": op, "of": members}
        if op in ("exactly", "at_least", "at_most"):
            pred["count"] = count if count is not None else 1
        return pred
    # Bare enumeration: "has a crest and a tail" → needs all of them.
    members = _enumerate(obj)
    if len(members) >= 2:
        return {"op": "all_of", "of": members}
    if len(members) == 1:
        return {"op": "any_of", "of": members}
    return None


def _attrs_from_object(obj: str, negated: bool) -> list[str] | None:
    """Item-description object → affirmed attribute names, or None
    when the statement only excludes ("doesn't have a dial" says
    nothing about what IS present — an under-described item the
    spec must not fabricate an empty attribute set for).

    "has nothing" is different — an explicit empty set, so it
    returns []. Negation with a "but" clause affirms only what
    follows it: "has no lens but a dial" → [dial].
    """
    low = obj.lower()
    if re.search(r"\b(nothing|none|naught)\b", low):
        rest = low.split(" but ", 1)[1] if " but " in low else ""
        return _enumerate(rest)
    if negated or re.search(r"\b(no|not|without|lacks?|n't)\b", low):
        if " but " in low:
            return _enumerate(low.split(" but ", 1)[1])
        return None
    affirmed = _enumerate(low)
    if re.search(r"\b(all|every|each|both|everything)\b", low) and (
        not affirmed or all(a == "__all__" or a.isdigit() for a in affirmed)
    ):
        # "has all three / everything" — the whole vocabulary; the
        # caller expands the sentinel once item attrs are collected.
        return ["__all__"]
    return affirmed


# ── Relation parsing ────────────────────────────────────────────
# Relation words → the world's canonical relations. Lexical data —
# the senses map surface words to the same spatial meaning, so any
# phrasing the parser normalizes here stays semantic.
_REL_SENSES: dict[str, str] = {
    "left": "left_of", "right": "right_of",
    "next": "next_to", "beside": "next_to", "adjacent": "next_to",
    "near": "next_to", "by": "next_to", "close": "next_to",
    "far": "apart", "apart": "apart", "away": "apart",
}
_REL_COMPLETE = frozenset({"of", "to", "from"})


def _rel_chunks(text: str) -> list[tuple[str, str]]:
    """A post-verbal NP string → alternating (np | rel | and) chunks.

    "the star left of the moon and the moon next to the sun" →
    [np star, rel left_of, np moon, and, np moon, rel next_to,
    np sun]. Relation words absorb their completing preposition.
    """
    toks = _tokens(text)
    chunks: list[tuple[str, str]] = []
    i = 0
    np_words: list[str] = []
    while i < len(toks):
        w = toks[i]
        if w in ("and", "or"):
            if np_words:
                name = _np_name(" ".join(np_words))
                if name:
                    chunks.append(("np", name))
                np_words = []
            chunks.append(("and", ""))
            i += 1
            continue
        if w in _REL_SENSES and i + 1 < len(toks) and (
            toks[i + 1] in _REL_COMPLETE or w in ("beside", "near", "by")
        ):
            if np_words:
                name = _np_name(" ".join(np_words))
                if name:
                    chunks.append(("np", name))
                np_words = []
            j = i + 1
            if j < len(toks) and toks[j] in _REL_COMPLETE:
                j += 1
            chunks.append(("rel", _REL_SENSES[w]))
            i = j
            continue
        np_words.append(w)
        i += 1
    if np_words:
        name = _np_name(" ".join(np_words))
        if name:
            chunks.append(("np", name))
    return chunks


def _rel_triples(chunks: list[tuple[str, str]]) -> list[tuple[str, str, str]]:
    """Fold chunks into (a, rel, b) triples. Pending NPs share the
    next relation ("the star and the moon left of the sun" applies
    left_of to both)."""
    triples: list[tuple[str, str, str]] = []
    pending: list[str] = []
    i = 0
    while i < len(chunks):
        kind, val = chunks[i]
        if kind == "np":
            pending.append(val)
            i += 1
            continue
        if kind == "and":
            i += 1
            continue
        # rel: pairs every pending NP with the following NP.
        if i + 1 < len(chunks) and chunks[i + 1][0] == "np":
            b = chunks[i + 1][1]
            for a in pending:
                if a != b:
                    triples.append((a, val, b))
            pending = []
            i += 2
            continue
        i += 1
    return triples


# ── Frame → task-family compilers ───────────────────────────────

def _classification_spec(result: ComprehensionResult) -> dict | None:
    """Assemble a rule-membership game from semantic frames.

    Indefinite-subject ISA propositions declare kinds; indefinite
    CONTAINS propositions on the kind (or its superordinate) carry
    the rule; definite CONTAINS propositions describe items; definite
    ISA propositions onto a kind assert labels; a wh-question over
    the plural kind is the query.
    """
    # Pass 1 — kind declarations: "a tessel is a machine", "wugs are
    # birds".
    kinds: dict[str, str] = {}  # kind → superordinate head
    supers: dict[str, str] = {}  # superordinate head → kind
    for p in result.propositions:
        if p.predicate not in _COPULA or not p.object or p.negated:
            continue
        if not _is_indefinite(p.subject):
            continue
        kind = _deplural(_np_name(p.subject))
        sup = _deplural(_np_name(p.object))
        if kind and sup and kind != sup:
            kinds[kind] = sup
            supers.setdefault(sup, kind)
    if not kinds:
        return None

    resolved = getattr(result, "resolved_references", None) or {}

    def _entity_name(subject: str) -> str:
        """Pronoun subjects resolve through comprehension's
        reference map; definite NPs normalize directly."""
        s = subject.strip().lower()
        if s in resolved:
            return _np_name(resolved[s])
        return _np_name(subject)

    # Pass 2 — rule bodies (indefinite CONTAINS) and items/labels
    # (definite CONTAINS / ISA onto a kind).
    pred_raw_obj: str | None = None
    items: list[dict[str, Any]] = []
    seen_items: set[str] = set()
    asserted: dict[str, bool] = {}
    for p in result.propositions:
        pred = p.predicate.lower()
        obj = p.object or ""
        # "doesn't have a dial" — do-support leaves the lexical verb
        # inside the object.
        if pred in ("do", "does", "did") and p.negated:
            rest = re.sub(r"^(?:not|n't|never)\s+", "", obj.lower())
            if rest.split()[:1] and rest.split()[0] in _CONTAINS:
                pred = rest.split()[0]
                obj = " ".join(rest.split()[1:])
        if pred in _CONTAINS and obj:
            if _is_indefinite(p.subject):
                # Rule body — the subject is the kind or its
                # superordinate ("a machine has exactly two of...").
                head = _deplural(_np_name(p.subject))
                if head in kinds or head in supers or not pred_raw_obj:
                    pred_raw_obj = obj
                continue
            name = _entity_name(p.subject)
            if not name or name in _NON_ENTITY or name in kinds:
                continue
            attrs = _attrs_from_object(obj, p.negated)
            if attrs is None:
                # Pure exclusion — nothing affirmed; under-described.
                continue
            if "__all__" in attrs:
                # "has all three / everything" — expands to the
                # attribute vocabulary once the loop collects it.
                attrs = ["__all__"]
            if name in seen_items:
                for it in items:
                    if it["name"] == name:
                        it["has"] = sorted(set(it["has"]) | set(attrs))
                        break
            else:
                seen_items.add(name)
                items.append({"name": name, "has": attrs})
            continue
        if pred in _COPULA and obj:
            name = _entity_name(p.subject)
            target = _deplural(_np_name(obj))
            if (
                name
                and name not in _NON_ENTITY
                and not _is_indefinite(p.subject)
                and target in kinds
            ):
                asserted[name] = not p.negated
    if pred_raw_obj is None or not items:
        return None

    # The query: a question naming the kind ("which are tessels"),
    # or asserted labels to check.
    query = result.is_question and any(
        _deplural(t) in kinds
        for p in result.propositions
        for t in _tokens(f"{p.subject} {p.object}")
    )
    if not query and not asserted:
        return None

    # "has all three / everything" items expand to the attribute
    # vocabulary — which is the union of the items' concrete attrs
    # AND the rule's own enumerated set ("at least two of: a spin,
    # a charge, a color" names members no item may show). A bare
    # "these parts" rule reference resolves to that vocabulary too.
    vocab = sorted({a for it in items for a in it["has"] if a != "__all__"})
    pred_spec = _predicate_from_object(pred_raw_obj, vocab)
    if pred_spec is None:
        return None
    vocab = sorted(set(vocab) | set(pred_spec.get("of") or []))
    for it in items:
        if "__all__" in it["has"]:
            it["has"] = list(vocab)
    for name, label in asserted.items():
        for it in items:
            if it["name"] == name:
                it["label"] = label
    kind = next(iter(kinds))
    spec: dict[str, Any] = {
        "family": "classification",
        "kind": kind,
        "predicate": pred_spec,
        "items": items,
    }
    spec["name"] = _name("classification", spec)
    spec["hint"] = result.raw_text.strip()[:240]
    return spec


def _minimal_period(run: list[str]) -> int:
    """Smallest period the whole run is consistent with, else 0."""
    for p in range(1, len(run)):
        if all(run[i] == run[i % p] for i in range(len(run))):
            return p
    return 0


# Words that may precede the marks in a role phrase ("in a b a b",
# "the pattern goes a b a b") — stripped from the head only.
_SEQ_LEAD_IN = {
    "in", "on", "of", "the", "my", "this", "pattern", "sequence",
    "series", "list", "line", "row", "string", "goes", "is", "are",
    "like", "next", "comes",
}
_SEQ_GLUE = {"and", "or", "then"}


def _sequence_spec(result: ComprehensionResult) -> dict | None:
    """A continuation predicate over a mark run → sequence spec."""
    if not (result.is_question or result.is_command):
        return None
    has_predict = any(
        p.predicate.lower() in _PREDICT
        or "next" in _tokens(p.object)
        for p in result.propositions
    )
    if not has_predict:
        return None
    # Candidate mark runs live in the roles/object strings the
    # parser attached to the continuation verb.
    candidates: list[str] = []
    for p in result.propositions:
        if p.object:
            candidates.append(p.object)
        candidates.extend(str(v) for v in p.roles.values())
    best: list[str] = []
    for cand in candidates:
        toks = _tokens(cand)
        while toks and toks[0] in _SEQ_LEAD_IN:
            toks.pop(0)
        toks = [t for t in toks if t not in ("what", "which")]
        if len(toks) >= 3 and len(toks) > len(best):
            best = toks
    if not best:
        return None
    run = best
    period = _minimal_period(run)
    if period == 0:
        cleaned = [t for t in run if t not in _SEQ_GLUE]
        if 3 <= len(cleaned) < len(run):
            period = _minimal_period(cleaned)
            if period:
                run = cleaned
    if period == 0:
        return None
    run = run[-23:]
    spec: dict[str, Any] = {
        "family": "sequence",
        "pattern": run[:period],
        "length": len(run) + 1,
        "scaffold": len(run),
        "decoys": 2,
    }
    spec["name"] = _name("sequence", spec)
    spec["hint"] = result.raw_text.strip()[:240]
    return spec


def _relations_spec(result: ComprehensionResult) -> dict | None:
    """NP–relation–NP chunks → arrangement goals."""
    triples: list[tuple[str, str, str]] = []
    imperative = False
    for p in result.propositions:
        pred = p.predicate.lower()
        if pred in _PLACE:
            imperative = True
        # Placement imperatives put THEME+locations in the object;
        # copular/locative verbs put the relation phrase there with
        # the subject as the located thing ("the star IS left of
        # the moon"). Either way the field chunks the same.
        if p.object and (
            pred in _PLACE or pred in _COPULA or pred in _LOCATIVE
        ):
            field = f"{p.subject} {p.object}".strip()
            triples.extend(_rel_triples(_rel_chunks(field)))
    # Deduplicate while keeping order.
    seen: set[tuple[str, str, str]] = set()
    goals: list[dict] = []
    objects: list[str] = []
    for a, rel, b in triples:
        if a in _NON_ENTITY or b in _NON_ENTITY or a == b:
            continue
        if (a, rel, b) in seen:
            continue
        seen.add((a, rel, b))
        goals.append({"rel": rel, "a": a, "b": b})
        for n in (a, b):
            if n not in objects:
                objects.append(n)
    if not goals or len(objects) < 2:
        return None
    # A lone relational statement is more likely scene description
    # than a task; require an imperative or a second relation.
    if len(goals) < 2 and not (imperative or result.is_command):
        return None
    spec: dict[str, Any] = {
        "family": "relations",
        "positions": max(len(objects), 2),
        "objects": objects,
        "goals": goals,
    }
    spec["name"] = _name("relations", spec)
    spec["hint"] = result.raw_text.strip()[:240]
    return spec


def _quantities_spec(result: ComprehensionResult) -> dict | None:
    """An achieve-verb with numeric object + counted instrument.

    No speech-act gate — "make 9" may parse as a statement when the
    verb isn't in comprehension's imperative set; the frame itself
    (achieve + numeric object + ≥2 counted groups) is specific
    enough.
    """
    for p in result.propositions:
        if p.predicate.lower() not in _ACHIEVE:
            continue
        target = _num(p.object.split()[0]) if p.object else None
        if target is None or target < 1:
            continue
        # Counts arrive inside role phrases ("with groups of 2, 5
        # and 4" → INSTRUMENT) or in adjunct clauses the parser split
        # off ("reach 12 USING groups of 3, 4 and 5" → a 'using'
        # gerund proposition). Harvest numbers everywhere except the
        # target's own mention.
        counts: set[int] = set()
        fields: list[str] = [str(v) for v in p.roles.values()]
        fields.extend(p.object.split()[1:])
        for q in result.propositions:
            if q is not p:
                fields.append(q.object or "")
                fields.extend(str(v) for v in q.roles.values())
        for f in fields:
            for tok in _tokens(f):
                n = _num(tok)
                if n is not None and n >= 1 and n != target:
                    counts.add(n)
        if len(counts) < 2:
            continue
        groups = [
            {"name": f"group_of_{c}_{i}", "count": c}
            for i, c in enumerate(sorted(counts))
        ]
        spec: dict[str, Any] = {
            "family": "quantities",
            "target": target,
            "groups": groups,
        }
        spec["name"] = _name("quantities", spec)
        spec["hint"] = result.raw_text.strip()[:240]
        return spec
    return None


# ── Sorter frame ────────────────────────────────────────────────
# The shape sorter is conjunctive attribute matching: a slot named
# "the round hole" accepts blocks that HAVE round. Attributes are
# flat presence features — without an ontology there is no honest
# way to know "red" is a color, so accepts={"red"} matches any
# block that mentions red.
# Nouns that head a slot phrase ("a round hole", "the star slot").
_SLOT_NOUNS = frozenset({
    "hole", "holes", "slot", "slots", "opening", "openings",
    "aperture", "apertures", "socket", "sockets", "gap", "gaps",
})
# Filler nouns that head a block phrase ("a red peg") — the words
# are category labels, not attributes, so they are never criteria.
_FILLER_NOUNS = _CATEGORY_NOUNS | frozenset({
    "block", "blocks", "peg", "pegs", "shape", "shapes", "piece",
    "pieces", "token", "tokens", "tile", "tiles",
})
# "the star hole takes a small star" — the object spells the slot's
# acceptance criterion.
_ACCEPTS = frozenset({"takes", "take", "accepts", "accept", "accepting"})
# "a small star fits the star hole" — asserts a match the other
# direction: registers the block without widening the criterion.
_FIT = frozenset({"fits", "fit", "matches", "match", "suits", "suit"})
# An imperative that turns a described sorter into a task.
_SORT_DO = _PLACE | _ACCEPTS | frozenset({
    "sort", "drop", "insert", "fill", "slide",
})


def _np_attrs(name: str, heads: frozenset[str]) -> set[str]:
    """NP name → content words, minus head nouns and category filler.

    "small_red_star" → {small, red, star}; "round_hole" with the
    slot heads removed → {round}. Pronouns and task verbs inside a
    mangled NP ("its_hole", "sort_them") are never attributes.
    """
    return {
        t
        for t in name.split("_")
        if t
        and t not in heads
        and t not in _CATEGORY_NOUNS
        and t not in _NON_ENTITY
        and t not in _SORT_DO
        and t not in _ACCEPTS
        and t not in _FIT
    }


def _np_is_slot(name: str) -> bool:
    """The name's head noun is a slot word ("star_hole", "square_slot")."""
    parts = name.split("_")
    return bool(parts) and _deplural(parts[-1]) in _SLOT_NOUNS


def _sorter_spec(result: ComprehensionResult) -> dict | None:
    """A described shape sorter → task spec.

    Frames: "the box has a round hole and a square hole" declares
    slots under a containment verb; "there is a red round block"
    declares pieces through the existential; "it comes with a small
    star and a big moon" lists them as instruments; "the star hole
    takes a small star" states a criterion; "a small star fits the
    star hole" registers a block; "put each block in its hole" makes
    it a task. A scene with slots and pieces but no task signal —
    no imperative, no acceptance claim, no sorter vocabulary — is
    description, not a problem.
    """
    slots: dict[str, set[str]] = {}
    blocks: dict[str, set[str]] = {}
    imperative = False
    linked = False

    def slot_name(member: str) -> str | None:
        name = _np_name(member)
        return name if _np_is_slot(name) else None

    def add_slot(member: str) -> None:
        name = slot_name(member)
        if name is None:
            return
        attrs = _np_attrs(name, _SLOT_NOUNS)
        if attrs:
            slots.setdefault(name, set()).update(attrs)

    def add_block(member: str) -> None:
        name = _np_name(member)
        if not name or _np_is_slot(name) or name in _NON_ENTITY:
            return
        attrs = _np_attrs(name, _FILLER_NOUNS)
        if attrs:
            blocks.setdefault(name, set()).update(attrs)

    def members(field: str) -> list[str]:
        # "two holes: a round hole and a square hole" — the part
        # before the colon is a counted category mention, not a
        # member.
        if ":" in field:
            field = field.split(":", 1)[1]
        return _enumerate(field)

    for p in result.propositions:
        pred = p.predicate.lower()
        obj = p.object or ""
        if pred in _SORT_DO:
            imperative = True
        if pred in _ACCEPTS and obj:
            sname = slot_name(p.subject)
            if sname is not None:
                linked = True
                attrs = slots.setdefault(
                    sname, _np_attrs(sname, _SLOT_NOUNS)
                )
                for m in members(obj):
                    mname = _np_name(m)
                    attrs.update(_np_attrs(mname, _FILLER_NOUNS))
                    add_block(m)
            continue
        if pred in _FIT and obj:
            sname = slot_name(obj)
            if sname is not None:
                linked = True
                slots.setdefault(sname, _np_attrs(sname, _SLOT_NOUNS))
            for m in members(p.subject):
                add_block(m)
            continue
        if pred in _CONTAINS and obj:
            for m in members(obj):
                add_slot(m)
            for m in members(obj):
                if slot_name(m) is None:
                    add_block(m)
            continue
        # Existential or fragment props ("a red round block" is,
        # "a blue square" block") — the subject enumerates loose
        # pieces.
        if (pred in _COPULA or pred in _FILLER_NOUNS) and not obj:
            for m in members(p.subject):
                add_block(m)
        # Imperatives name pieces as objects and slots as
        # destinations ("put the red star in the square hole").
        if pred in _SORT_DO:
            for m in members(obj):
                add_block(m)
            for v in p.roles.values():
                for m in members(str(v)):
                    add_slot(m)
        # Pieces arriving as instruments ("it comes with a star").
        for role, v in p.roles.items():
            if "instrument" in role.name.lower():
                for m in members(str(v)):
                    add_block(m)

    # "sort them", "the shape sorter", "a sorting toy" — sorter
    # vocabulary in the raw text is itself a task signal.
    sorter_word = any(
        w in {"sort", "sorts", "sorted", "sorting", "sorter", "sorters"}
        for w in _tokens(result.raw_text)
    )
    if not slots or not blocks or not (imperative or linked or sorter_word):
        return None
    lid = not any(
        "lid" in (p.subject + " " + (p.object or "")).lower()
        and p.negated
        for p in result.propositions
    )
    spec: dict[str, Any] = {
        "family": "sorter",
        "slots": [
            {"accepts": dict.fromkeys(sorted(attrs), "yes")}
            for attrs in slots.values()
        ],
        "blocks": [
            {"attrs": dict.fromkeys(sorted(attrs), "yes")}
            for attrs in blocks.values()
        ],
        "lid": lid,
        # A heard sorter is a physical object — the described attrs
        # are its physics, but the agent works it by sight through
        # the visual cortex when one is wired in.
        "perceptual": True,
    }
    spec["name"] = _name("sorter", spec)
    spec["hint"] = result.raw_text.strip()[:240]
    return spec


# ── Entry point ─────────────────────────────────────────────────

_EXTRACTORS = (
    _classification_spec,
    _sequence_spec,
    _relations_spec,
    _quantities_spec,
    _sorter_spec,
)


def interpret_problem(result: ComprehensionResult) -> dict | None:
    """Interpret a comprehension result as a task spec, or None.

    Every extractor works on semantic structure — propositions,
    roles, negation, speech act — never on raw word order. A spec is
    returned only when the frames assemble into a complete,
    verifiable task; ``normalize_offered`` validates solvability and
    consistency before it becomes one.
    """
    if not result.propositions or len(result.raw_text) > 4000:
        return None
    for extractor in _EXTRACTORS:
        try:
            spec = extractor(result)
        except Exception:  # noqa: BLE001 — intake never crashes think()
            spec = None
        if spec is not None:
            return spec
    return None


def compile_problem(
    text: str,
    comprehension: ComprehensionEngine | None = None,
) -> dict | None:
    """Convenience wrapper: comprehend raw text, then interpret.

    The real entry point is ``interpret_problem`` — cognition passes
    the ``ComprehensionResult`` it already computed. This wrapper
    exists for tests and tools that only have the text.
    """
    if not text or not text.strip():
        return None
    if comprehension is None:
        from ..language.comprehension import ComprehensionEngine

        comprehension = ComprehensionEngine()
    try:
        return interpret_problem(comprehension.comprehend(text))
    except Exception:  # noqa: BLE001 — intake never crashes think()
        return None
