"""Situation model — the state of the world the current discourse describes.

Distinct from the concept network: the network holds semantic knowledge
("a tardigrade is a microscopic animal") while this holds the episodic
situation being talked about right now ("Mary is in the bathroom",
"Sandra is carrying the milk"). Propositions from the comprehension
engine update it every turn; question answering consults it before
falling back to semantic memory or external lookup — so a "Where is
Mary?" asked mid-story is answered from what she just heard, not from
the web.

Modeled on the discourse situation model of van Dijk & Kintsch (1983):
comprehension does not end at the sentence — it maintains a running
model of who is where, holding what, in what relation to whom.

What it tracks per entity:
    - location (latest wins) plus a history for "where was X before Y"
    - negated locations ("John is not in the hallway")
    - uncertain locations ("either in the garden or the bathroom",
      "might be in the office")
    - possessions (take/drop/give) — carried objects co-locate with
      their holder, including through the holder's moves
    - attributes and class membership ("a mouse", "hungry", "afraid
      of cats") — for deduction/induction and motivation questions
    - place relations ("the kitchen is north of the hallway") for
      relational and path-finding questions
    - size/order relations ("fits in", "bigger than") for size
      reasoning

Event-indexed structure (Zwaan & Radvansky, 1998):
    Every proposition also becomes a :class:`SituationEvent` — a bound
    relational unit indexed on the five dimensions readers use to
    track a situation:

    - protagonist: every participant in the event (``by_participant``)
    - space: where the event happens (``by_location``)
    - time: sequence, tense, and explicit temporal phrases
    - causation: stated reasons ("because ...") and consequence
      markers ("so", "therefore") link cause → effect events
    - intention: purpose clauses ("to buy milk") and, separately,
      the interlocutor's questions/commands (``requests``)

    Events bind to each other by shared participants and location
    (construction–integration; Kintsch, 1988). An event that binds to
    nothing in the active window marks an episode boundary — the
    discourse has shifted to a new situation.

Attribution and scope:
    Embedded propositions (comprehension's ``Proposition.embedded``)
    are *attributed* content, never asserted. "John believes that
    Mary is in the kitchen" logs a belief event about John and stores
    "mary is in the kitchen" under John's ``beliefs`` — it does not
    move Mary. Mental-state verbs (think, believe, hope, ...) write to
    the belief index (``beliefs_of``); report verbs (say, tell, ...)
    stay on the event log as public speech acts (``reports_of``).
    Embedded entities join the event's participant index so the
    attribution binds into the discourse without becoming world state.

Only statements update entity state and the event log. Questions and
commands are recorded as dialogue acts (the interlocutor's expressed
intentions) but never mutate the state being talked about.
"""

from __future__ import annotations

import re
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from ..language.morphology import deconjugate_verb, is_verb_form

if TYPE_CHECKING:
    from ..language.comprehension import ComprehensionResult, Proposition

_ARTICLE_RE = re.compile(r"^(?:the|a|an)\s+")
_PREP_RE = re.compile(
    r"^(?:in|inside|at|on|into|onto|within|near|by|to|back to)\s+"
)
_UNCERTAIN_RE = re.compile(r"\b(?:might|maybe|perhaps|either)\b", re.I)
_AND_RE = re.compile(r"\s+(?:and|or)\s+")
# A stated reason: "went to the kitchen because she was hungry".
_CAUSE_RE = re.compile(r"\bbecause\s+(.+?)\s*$", re.I)
# Discourse markers glued onto the front of a clause ("Therefore she",
# "Then he") — structure, not part of the subject.
_DISCOURSE_RE = re.compile(
    r"^(?:therefore|thus|hence|then|now|next|however|"
    r"afterwards?|later|meanwhile)\b[,]?\s*",
    re.I,
)
# Explicit purpose markers beyond a bare "to <verb>" infinitive.
_PURPOSE_RE = re.compile(
    r"\b(?:in order to|so as to)\s+(.+?)\s*$", re.I
)

# Attributed-belief memory is small like the rest of the model — an
# entity's recent beliefs, not an encyclopedia.
_MAX_BELIEFS = 8

# Verb families, as lemmas and common past forms — the vocabulary the
# situation model understands. These are building blocks (seeds), not
# responses.
_MOTION = frozenset({
    "go", "went", "gone", "move", "moved", "journey", "journeyed",
    "travel", "travelled", "traveled", "run", "ran", "walk", "walked",
    "march", "marched", "flee", "fled", "head", "headed", "return",
    "returned", "come", "came", "leave", "left for", "enter", "entered",
    "get", "sneak", "sneaked", "hurry", "hurried", "dash", "dashed",
})
_TAKE = frozenset({
    "take", "took", "grab", "grabbed", "get", "got", "pick", "picked",
    "hold", "held", "carry", "carried", "snatch", "snatched",
    "retrieve", "retrieved",
})
_DROP = frozenset({
    "drop", "dropped", "leave", "left for", "discard", "discarded",
    "put down", "put", "set down", "abandon", "abandoned",
})
# A possession verb in a copula frame: comprehension yields
# "Sandra is carrying the milk" as predicate "is" with the whole
# verb phrase in ATTRIBUTE, so the verb arrives here as a participle
# ("carrying", "held") rather than the lemma the _TAKE/_DROP/_GIVE
# sets hold ("carry", "hold"). Every inflection of those families is
# matched explicitly — stemming is not worth the ambiguity here, and a
# missed form only costs the possession answer.
_POSSESSIVE_VERBS = (
    "carrying", "carry", "carried",
    "holding", "hold", "held",
    "taking", "take", "took", "taken",
    "grabbing", "grab", "grabbed",
    "picking", "pick", "picked",
    "snatching", "snatch", "snatched",
    "retrieving", "retrieve", "retrieved",
    "dropping", "drop", "dropped",
    "discarding", "discard", "discarded",
    "abandoning", "abandon", "abandoned",
    "putting down", "put down", "setting down", "set down",
    "giving", "give", "gave", "given",
    "handing", "hand", "handed",
    "passing", "pass", "passed",
    "offering", "offer", "offered",
)
# Longest-first so "putting down" wins over "putting".
_GERUND_POSSESSION_RE = re.compile(
    r"^(" + "|".join(sorted(_POSSESSIVE_VERBS, key=len, reverse=True)) + r")\s+"
    r"(?:the\s+|a\s+|an\s+)?(.+?)$",
    re.I,
)
# Present participle → the lemma the _TAKE/_DROP/_GIVE sets actually
# hold. Those sets contain base and past forms only, so matching the
# participle and then testing membership requires this map; matching
# the participle and returning it would never hit, which is why the
# family test below compares against the normalized verb.
_PARTICIPLE_LEMMA = {
    "carrying": "carry", "holding": "hold", "taking": "take",
    "grabbing": "grab", "picking": "pick", "snatching": "snatch",
    "retrieving": "retrieve", "dropping": "drop",
    "discarding": "discard", "abandoning": "abandon",
    "putting down": "put down", "setting down": "set down",
    "giving": "give", "handing": "hand", "passing": "pass",
    "offering": "offer", "taken": "take",
}
_GIVE = frozenset({
    "give", "gave", "hand", "handed", "pass", "passed", "offer",
    "offered",
})
_COPULA = frozenset({
    "is", "are", "was", "were", "be", "being", "been",
    "isn't", "aren't", "wasn't", "weren't",
})

_DIRS = frozenset({"north", "south", "east", "west"})
_POS = frozenset({"left", "right", "above", "below"})
_DIR_OF_RE = re.compile(
    r"\b(north|south|east|west)\s+of\s+(?:the\s+)?(.+?)\s*$", re.I
)
_POS_OF_RE = re.compile(
    r"\b(?:to\s+the\s+)?(left|right)\s+of\s+(?:the\s+)?(.+?)\s*$"
    r"|\b(above|below)\s+(?:the\s+)?(.+?)\s*$",
    re.I,
)
_AFRAID_RE = re.compile(r"\bafraid\s+of\s+(?:the\s+)?(.+?)\s*$", re.I)
_FITS_RE = re.compile(r"(.+?)\s+fits?\s+in\s+(?:the\s+)?(.+?)\s*$", re.I)
_BIGGER_RE = re.compile(
    r"(.+?)\s+(?:is|are|was|were)\s+(bigger|larger|smaller)\s+than\s+"
    r"(?:the\s+)?(.+?)\s*$", re.I
)
_SIZE_ATTR = {"big": 1, "large": 1, "small": -1, "little": -1, "tiny": -1}
_STATE_WORDS = frozenset({
    "hungry", "thirsty", "tired", "sleepy", "bored", "afraid", "scared",
    "happy", "sad", "angry", "full", "cold", "hot", "sick",
})
_COLORS = frozenset({
    "white", "black", "red", "green", "blue", "yellow", "gray", "grey",
    "brown", "orange", "pink", "purple",
})
# Irregular plurals seen in this vocabulary domain.
_PLURALS = {"mice": "mouse", "geese": "goose", "men": "man",
            "women": "woman", "children": "child", "sheep": "sheep",
            "wolves": "wolf"}


def _norm(text: str) -> str:
    """Normalize an entity/place name: lowercase, strip article."""
    return _ARTICLE_RE.sub("", text.strip().lower().rstrip("?.,!"))


def _place(text: str) -> str:
    """Normalize a location phrase: strip preposition + article."""
    return _norm(_PREP_RE.sub("", text.strip().lower()))


def _entities(text: str) -> list[str]:
    """Split a (possibly conjoined) subject into entity names."""
    return [e for e in (_norm(p) for p in _AND_RE.split(text)) if e]


def _split_cause(text: str) -> tuple[str, str]:
    """Split a trailing "because <clause>" off a phrase.

    The parser keeps "because" inside one clause, so a reason clause
    can glue onto a goal/location/object ("the kitchen because she
    was hungry"). The head is the actual phrase; the tail is the
    stated cause — indexed separately as causation, never as a place
    or entity.
    """
    m = _CAUSE_RE.search(text)
    if not m:
        return text, ""
    return text[: m.start()].strip(), m.group(1).strip()


def _is_purpose(phrase: str) -> bool:
    """Whether a "to <phrase>" goal is a purpose infinitive.

    "to buy milk" is an intention (the head is a verb); "to the
    kitchen" is a destination (the head is a noun phrase). Detected
    morphologically via the shared lexicon. A single-word verb
    ("went to work") defaults to destination — bare-NP places like
    work/school are the more common reading after a motion verb.
    """
    words = phrase.split()
    if len(words) < 2:
        return False
    return is_verb_form(words[0]) is not None


def _destination_before(raw: str, purpose_head: str) -> str:
    """Recover a "to <NP>" destination stated before a purpose clause.

    The parser's GOAL role keeps only the last "to"-phrase, so in
    "went to the store to buy milk" the destination is overwritten
    by the purpose. Recover any to-phrase that precedes it.
    """
    m = re.search(
        r"\bto\s+((?:(?:the|a|an)\s+)?\w+(?:\s+\w+)*?)\s+to\s+"
        + re.escape(purpose_head)
        + r"\b",
        raw,
        re.I,
    )
    return m.group(1) if m else ""


def _verb_match(asked: str, logged: str) -> bool:
    """Whether an asked-about verb matches a logged event predicate.

    Morphological first ("go" matches "went" via shared base forms),
    then verb-family: the event log normalizes surface verbs into
    state transitions — "went" becomes "moved to" — so a question's
    "go" must match through the motion family, not just morphology.
    """
    a_bases = set(deconjugate_verb(asked))
    head = logged.split()[0] if logged else ""
    b_bases = set(deconjugate_verb(head))
    if a_bases & b_bases:
        return True
    for family in (_MOTION, _TAKE, _DROP, _GIVE, _COPULA):
        a_in = asked in family or bool(a_bases & family)
        b_in = head in family or bool(b_bases & family)
        if a_in and b_in:
            return True
    return False


# Propositional-attitude verbs — the mental-state subset of the
# complement-taking predicates. A believed/thought/hoped clause is
# attributed to the holder's beliefs; speech verbs (say, tell,
# report...) are public acts and stay on the event log only.
_ATTITUDE = frozenset({
    "think", "believe", "know", "hope", "expect", "suppose", "guess",
    "imagine", "assume", "suspect", "doubt", "fear", "wish", "want",
    "feel", "remember", "understand", "realize", "figure", "conclude",
    "dream", "wonder", "decide",
})

# Report predicates — public speech acts whose complement is what was
# said, not what is believed. "Mary said that P" attributes P to
# Mary's utterance.
_CTP_REPORT = frozenset({
    "say", "tell", "report", "claim", "state", "describe", "mention",
    "admit", "confess", "insist", "maintain", "argue", "warn",
    "promise", "explain", "suggest", "imply", "announce", "whisper",
    "shout", "ask", "write", "read", "hear",
})

# Role → preposition for compact proposition descriptions.
_ROLE_PREP = (
    ("location", "in"), ("goal", "to"), ("time", "during"),
    ("theme", "about"), ("beneficiary", "for"), ("instrument", "with"),
    ("source", "from"),
)


def _describe_prop(prop: Proposition) -> str:
    """Render a proposition as compact content text.

    Used for embedded clause content — "mary is in the kitchen" —
    as the object of a belief/report event or a belief entry. This
    is a *description of content*, not asserted state and not her
    spoken words; it exists so attribution can carry structure.
    """
    parts = [
        p
        for p in (_norm(prop.subject), prop.predicate.lower(), _norm(prop.object))
        if p
    ]
    if prop.negated and len(parts) >= 2:
        parts.insert(2, "not")
    for role_label, prep in _ROLE_PREP:
        value = ""
        for role, filler in prop.roles.items():
            if role.label == role_label:
                value = filler
                break
        if value:
            parts.append(f"{prep} {value}")
    if prop.embedded is not None:
        parts.append(f"that {_describe_prop(prop.embedded)}")
    return " ".join(parts)


@dataclass
class EntityState:
    """The discourse state of one entity."""

    location: str | None = None
    location_seq: int = -1               # event index of the latest location
    location_history: list[tuple[int, str]] = field(default_factory=list)
    not_at: set[str] = field(default_factory=set)
    candidates: set[str] = field(default_factory=set)
    carrying: list[str] = field(default_factory=list)
    holder: str = ""                     # for objects: who carries it
    classes: set[str] = field(default_factory=set)
    attributes: set[str] = field(default_factory=set)
    fears: set[str] = field(default_factory=set)
    #: Attributed beliefs — "John believes that Mary is in the
    #: kitchen" stores the embedded content here, under John, without
    #: asserting it. An entity's model of the world, not the world.
    beliefs: list[str] = field(default_factory=list)


@dataclass
class SituationEvent:
    """One bound event — a clause's contribution to the situation.

    The unit of the event-indexed model (Zwaan & Radvansky, 1998):
    each event is indexed on the five dimensions a comprehender uses
    to keep track of what is happening.

    Attributes:
        seq: Temporal position — the event's index in the log.
        episode: Which episode this belongs to. A new episode starts
            when an event binds to nothing in the active window.
        subject: The protagonist/agent of the event (normalized).
        predicate: The relation or action.
        object: The patient/theme/place argument ("" when absent).
        participants: Every entity the event binds — subject, object
            parts, and clause co-subjects — ordered by mention.
        location: Spatial index — where the event situates.
        time_marker: Explicit temporal phrase ("during the storm").
        tense: "past"/"present"/"future"/"unknown" from the parse.
        negated: Whether the proposition was negated.
        uncertain: Whether it was hedged ("might", "either ... or").
        bound_to: Seq of prior events sharing a participant or the
            location — the construction–integration binding that makes
            the log a graph rather than a list (Kintsch, 1988).
        cause_text: Stated reason from a "because" clause, verbatim.
        causes: Seq of events this one is stated to cause/enable
            ("X so Y", "X. Therefore Y").
        caused_by: Seq of events stated to cause this one.
        intention: Stated purpose ("to buy milk", "in order to sleep").
    """

    seq: int
    episode: int
    subject: str
    predicate: str
    object: str
    participants: tuple[str, ...] = ()
    location: str = ""
    time_marker: str = ""
    tense: str = "unknown"
    negated: bool = False
    uncertain: bool = False
    bound_to: tuple[int, ...] = ()
    cause_text: str = ""
    causes: tuple[int, ...] = ()
    caused_by: tuple[int, ...] = ()
    intention: str = ""

    def describe(self) -> str:
        """Compact "subject predicate object" rendering."""
        return " ".join(
            p for p in (self.subject, self.predicate, self.object) if p
        )


@dataclass
class DialogueAct:
    """The interlocutor's expressed intention — a question or command.

    Statements describe the world; questions and commands express what
    the speaker wants. They are indexed as intentions of the
    interlocutor (Zwaan's motivation dimension, applied to the
    discourse itself) rather than as events in the described world —
    asking "Where is Mary?" must not move Mary anywhere.

    Attributes:
        seq: Request index.
        kind: "question" or "command".
        subject: What the act is about (normalized).
        predicate: The verb of the request ("tell", "is").
        object: Its object argument.
        focus: For questions, the questioned constituent ("where").
        theme: THEME role content ("tell me about X" → "x").
    """

    seq: int
    kind: str
    subject: str = ""
    predicate: str = ""
    object: str = ""
    focus: str = ""
    theme: str = ""


@dataclass
class _EventCtx:
    """Per-proposition context carried while a clause is applied.

    The role fillers after sanitization — cause clauses split off,
    purposes separated from destinations — plus the resolved subjects.
    ``_log`` reads it to build a fully-indexed SituationEvent from a
    bare (subject, predicate, object) call.
    """

    subjects: list[str]
    obj: str
    location: str
    destination: str
    intention: str
    cause_text: str
    time_marker: str
    tense: str
    negated: bool
    uncertain: bool
    #: Entities inside an embedded clause — referents that bind the
    #: attribution event to the discourse without becoming arguments
    #: of the matrix event.
    extra_participants: tuple[str, ...] = ()


class SituationModel:
    """Running model of the situation under discussion."""

    #: How many recent events a new event tries to bind to. Binding
    #: (shared participant or location) is what makes the model
    #: coherent; an unbound event opens a new episode.
    BIND_WINDOW: int = 8
    #: Interlocutor requests retained (the pending-intentions buffer).
    MAX_REQUESTS: int = 8

    def __init__(self) -> None:
        self.entities: dict[str, EntityState] = {}
        # place → direction → place ("kitchen"→"north"→"hallway" means
        # kitchen is north of the hallway)
        self.place_dir: dict[str, dict[str, str]] = defaultdict(dict)
        # positional relations: entity → relation → target
        self.pos_rel: dict[str, dict[str, str]] = defaultdict(dict)
        # partial size order: smaller → set of bigger
        self.smaller_than: dict[str, set[str]] = defaultdict(set)
        # ordered event log of bound SituationEvents
        self.events: list[SituationEvent] = []
        self._seq = 0
        # ── Event-indexing dimensions ──
        # protagonist index: entity → event seqs it participates in
        self.by_participant: dict[str, list[int]] = defaultdict(list)
        # space index: place → event seqs situated there
        self.by_location: dict[str, list[int]] = defaultdict(list)
        # episode segments: episode number → event seqs
        self.by_episode: dict[int, list[int]] = defaultdict(list)
        # intention index: the interlocutor's questions/commands
        self.requests: list[DialogueAct] = []
        self._req_seq = 0
        self._episode = 0
        # Set when the latest update crossed an episode boundary —
        # a discourse discontinuity signal for downstream consumers.
        self.episode_shifted = False
        # Per-proposition context while a clause is applied.
        self._ctx: _EventCtx | None = None

    # ─── Update ──────────────────────────────────────────────────

    def update(self, result: ComprehensionResult) -> None:
        """Apply a comprehended turn to the model.

        Statements mutate entity state and append bound events.
        Questions and commands are recorded as the interlocutor's
        intentions (``requests``) but never mutate described state.
        """
        uncertain = bool(_UNCERTAIN_RE.search(result.raw_text))
        if result.is_question or result.is_command:
            self._record_request(result)
            return
        self.episode_shifted = False
        start_seq = self._seq
        try:
            for prop in result.propositions:
                if prop.focus:  # interrogative residue — never a statement
                    continue
                self._apply(prop, result, uncertain)
        finally:
            self._ctx = None
        self._link_stated_causes(result, start_seq)

    def _entity(self, name: str) -> EntityState:
        return self.entities.setdefault(name, EntityState())

    def _subjects(
        self,
        prop: Proposition,
        raw: str,
        resolved: dict[str, str] | None = None,
    ) -> list[str]:
        """Entity names for this proposition's subject.

        Discourse markers are stripped ("Therefore she" → "she"),
        pronouns resolve through the comprehension result's reference
        map ("she" → "mary"), and the raw text's leading noun phrase
        is the fallback when the parse produced an empty subject.
        """
        subject = _DISCOURSE_RE.sub("", prop.subject.strip())
        if not subject:
            m = re.match(r"^\s*([A-Z][a-zA-Z]*(?:\s+(?:and|or)\s+[A-Z][a-zA-Z]*)*)", raw)
            if m:
                subject = _DISCOURSE_RE.sub("", m.group(1))
        resolved = resolved or {}
        out: list[str] = []
        for e in _entities(subject):
            referent = resolved.get(e.lower())
            name = _norm(referent) if referent else e
            # Irregular plurals name the class, not a distinct entity:
            # "Mice are afraid of cats" stores on 'mouse' so members of
            # the class inherit it.
            out.append(_PLURALS.get(name, name))
        return out

    @staticmethod
    def _walk(prop: Proposition):
        """Yield a proposition and every embedded proposition beneath it.

        Embedded content is attributed, not asserted — this walk exists
        for participant extraction and content description, never for
        state updates.
        """
        yield prop
        if prop.embedded is not None:
            yield from SituationModel._walk(prop.embedded)

    def _apply(self, prop: Proposition, result: ComprehensionResult,
             uncertain: bool) -> None:
        raw = result.raw_text
        pred = prop.predicate.lower()
        subjects = self._subjects(prop, raw, result.resolved_references)

        # Sanitize role fillers. "because" never splits a clause in
        # the parser, so a stated reason can glue onto a goal,
        # location, or object — split it off as causation before it
        # contaminates a place or entity name.
        goal, goal_cause = _split_cause(prop.roles.get(_ROLE_GOAL) or "")
        loc, loc_cause = _split_cause(prop.roles.get(_ROLE_LOCATION) or "")
        obj, obj_cause = _split_cause(prop.object.strip())
        attr = prop.roles.get(_ROLE_ATTRIBUTE)
        ben = prop.roles.get(_ROLE_BENEFICIARY)

        # Separate purposes from destinations. "to buy milk" is an
        # intention (infinitive); "to the kitchen" is a place. When a
        # purpose overwrote a stated destination in the GOAL role,
        # recover the earlier "to <NP>" from the raw text. Raw-text
        # extraction is clause-blind, so it only runs when the result
        # holds a single proposition — otherwise a sibling clause's
        # purpose could be misattributed to this one.
        destination = goal
        intention = ""
        single = len(result.propositions) <= 1
        purpose_m = _PURPOSE_RE.search(raw) if single else None
        if purpose_m:
            intention = purpose_m.group(1)
        elif goal and _is_purpose(goal):
            intention = goal
            if single:
                destination = _destination_before(raw, goal.split()[0])
            else:
                destination = ""

        time_marker, time_cause = _split_cause(
            prop.roles.get(_ROLE_TIME) or ""
        )
        self._ctx = _EventCtx(
            subjects=subjects,
            obj=obj,
            location=_place(loc) if loc else "",
            destination=_place(destination) if destination else "",
            intention=intention.strip(),
            cause_text=goal_cause or loc_cause or obj_cause or time_cause,
            time_marker=time_marker.strip(),
            tense=prop.tense,
            negated=prop.negated,
            uncertain=uncertain,
        )

        # ── Embedded clause: attributed content, not world state ──
        # "John believes that Mary is in the kitchen" asserts John's
        # believing — it must NOT move Mary. Log the attribution as
        # an event; for mental-state verbs, attach the content to
        # the holder's beliefs. This branch precedes every regex
        # branch below so embedded text ("north of", "afraid of")
        # can never contaminate described world state.
        if prop.embedded is not None:
            content = _describe_prop(prop.embedded)
            if prop.negated:
                content = f"not {content}"
            self._ctx.obj = content
            self._ctx.extra_participants = tuple(
                e
                for p in self._walk(prop.embedded)
                for e in (
                    _norm(p.subject),
                    _norm(p.object),
                    *(_norm(v) for v in p.roles.values()),
                )
                if e
            )
            base = is_verb_form(pred) or pred
            if base in _ATTITUDE:
                for s in subjects:
                    beliefs = self._entity(s).beliefs
                    if content not in beliefs:
                        beliefs.append(content)
                        del beliefs[:-_MAX_BELIEFS]
            self._log(subjects[0] if subjects else pred, pred, "")
            return

        # ── Relations expressed through copula/attributes ──
        dir_m = _DIR_OF_RE.search(raw)
        pos_m = _POS_OF_RE.search(raw)
        fits_m = _FITS_RE.search(raw)
        bigger_m = _BIGGER_RE.search(raw)
        afraid_m = _AFRAID_RE.search(raw)

        if dir_m and subjects:
            d, target = dir_m.group(1).lower(), _norm(dir_m.group(2))
            for s in subjects:
                self.place_dir[s][d] = target
            self._log(subjects[0], f"is {d} of", target)
            return
        if pos_m and subjects:
            rel = (pos_m.group(1) or pos_m.group(3) or "").lower()
            target = _norm(pos_m.group(2) or pos_m.group(4) or "")
            if rel and target:
                for s in subjects:
                    self.pos_rel[s][rel] = target
                # left/right and above/below are converse pairs —
                # store the inverse so "is circle left of square?"
                # answers no rather than unknown.
                self.pos_rel[target][_POS_OPPOSITE[rel]] = subjects[0]
                self._log(subjects[0], f"is {rel} of", target)
            return
        if fits_m:
            small, big = _norm(fits_m.group(1)), _norm(fits_m.group(2))
            if small and big and small != big:
                self.smaller_than[small].add(big)
                self._log(small, "fits in", big)
            return
        if bigger_m:
            a = _norm(bigger_m.group(1))
            b = _norm(bigger_m.group(3))
            if bigger_m.group(2) in ("bigger", "larger"):
                self.smaller_than[b].add(a)
            else:
                self.smaller_than[a].add(b)
            self._log(a, f"is {bigger_m.group(2)} than", b)
            return
        if afraid_m and subjects:
            feared = _norm(afraid_m.group(1))
            for s in subjects:
                self._entity(s).fears.add(feared)
            self._log(subjects[0], "is afraid of", feared)
            return

        # ── Motion: subject moves to the destination ──
        if pred in _MOTION and destination:
            for s in subjects:
                self._set_location(s, _place(destination), uncertain)
            return
        if pred in _MOTION and subjects:
            # "Where did X go?" needs the event even without a goal.
            self._log(subjects[0], pred, "")
            return

        # ── Possession ──
        if pred in _TAKE and obj:
            for s in subjects:
                self._take(s, _norm(obj))
            return
        if pred in _DROP and obj:
            for s in subjects:
                self._drop(s, _norm(obj))
            return
        if pred in _GIVE and (obj or ben):
            self._give(subjects, obj, ben)
            return

        # ── Copula: location, negation, class, attribute ──
        if pred in _COPULA:
            # A progressive or participial possession ("Sandra is
            # carrying the milk", "Mary is holding the cup") reaches
            # here as copula + ATTRIBUTE, not as a _TAKE predicate:
            # comprehension puts the whole verb phrase in the
            # attribute slot. Without this the verb never reaches the
            # possession branches below, and the object is recorded
            # as a bare attribute of Sandra instead of something she
            # holds — so "what is Sandra carrying?" has no answer.
            if self._apply_possessive_copula(subjects, attr or obj, uncertain):
                return
            self._apply_copula(subjects, prop, loc, attr, uncertain, raw)
            return

        # Anything else still lands in the event log.
        if subjects:
            self._log(subjects[0], pred, _norm(obj) if obj else "")

    def _apply_possessive_copula(
        self, subjects: list[str], attr_text: str, uncertain: bool
    ) -> bool:
        """Handle "X is carrying/holding Y" — possession via copula.

        Returns True when the attribute text was a possession verb and
        has been applied, so the caller skips the ordinary attribute
        path. The verb family is the same ``_TAKE`` / ``_DROP`` /
        ``_GIVE`` sets the direct-predicate branches use, matched
        morphologically because comprehension hands over the
        participle rather than the lemma ("carrying", "held",
        "carried").
        """
        if not attr_text or not subjects:
            return False
        text = attr_text.strip().lower()
        # "carrying the milk" → verb, remainder.
        m = _GERUND_POSSESSION_RE.match(text)
        if not m:
            return False
        verb, remainder = m.group(1).lower(), _norm(m.group(2))
        if not remainder:
            return False

        # Normalize the participle to the lemma the family sets hold
        # ("carrying" → "carry"). A verb already in a base or past form
        # passes through unchanged.
        phrase = _PARTICIPLE_LEMMA.get(verb, verb)

        if phrase in _GIVE:
            for s in subjects:
                self._give([s], remainder, None)
            return True
        if phrase in _DROP:
            for s in subjects:
                self._drop(s, remainder)
            return True
        if phrase in _TAKE:
            for s in subjects:
                self._take(s, remainder)
            return True
        return False

    def _apply_copula(self, subjects, prop, loc, attr, uncertain, raw) -> None:
        if loc:
            place = _place(loc)
            alternatives = [_norm(p) for p in _AND_RE.split(place) if p]
            if uncertain or len(alternatives) > 1:
                for s in subjects:
                    e = self._entity(s)
                    e.candidates.update(alternatives)
                self._log(subjects[0], "might be in", " or ".join(alternatives))
                return
            for s in subjects:
                if prop.negated:
                    e = self._entity(s)
                    e.not_at.add(place)
                    if e.location == place:
                        e.location = None
                    e.candidates.discard(place)
                    self._log(s, "is not in", place)
                else:
                    self._set_location(s, place, uncertain)
            return

        # "X is a <class>" / "X is <attribute>"
        obj = self._ctx.obj if self._ctx is not None else prop.object.strip()
        attr_text = (attr or obj)
        if not attr_text:
            return
        value = _norm(attr_text)
        if not value:
            return
        for s in subjects:
            e = self._entity(s)
            if prop.negated:
                e.attributes.discard(value)
                self._log(s, "is not", value)
                continue
            e.attributes.add(value)
            # "X is a mouse" is class membership (article + bare noun),
            # not a property — members query their class's attributes.
            if re.match(r"^(?:a|an)\s+\S+$", attr_text.strip().lower()):
                e.classes.add(_PLURALS.get(value, value))
            self._log(s, "is", value)

    # ─── State transitions ───────────────────────────────────────

    def _set_location(self, entity: str, place: str, uncertain: bool = False) -> None:
        e = self._entity(entity)
        if uncertain:
            e.candidates.add(place)
            return
        e.candidates.clear()
        if e.location == place:
            return
        e.location = place
        e.location_seq = self._seq
        e.location_history.append((self._seq, place))
        e.not_at.discard(place)
        self._log(entity, "moved to", place)
        # Carried objects go where the holder goes.
        for carried in e.carrying:
            self._set_location(carried, place)

    def _take(self, entity: str, obj: str) -> None:
        holder = self._entity(entity)
        if obj not in holder.carrying:
            holder.carrying.append(obj)
        o = self._entity(obj)
        o.holder = entity
        if holder.location:
            self._set_location(obj, holder.location)
        self._log(entity, "took", obj)

    def _drop(self, entity: str, obj: str) -> None:
        holder = self._entity(entity)
        if obj in holder.carrying:
            holder.carrying.remove(obj)
        o = self._entity(obj)
        o.holder = ""
        if holder.location:
            self._set_location(obj, holder.location)
        self._log(entity, "dropped", obj)

    def _give(self, subjects: list[str], obj: str, ben: str | None) -> None:
        """'John gave Mary the book' — object glues recipient + item."""
        if not subjects:
            return
        recipient = _norm(ben) if ben else ""
        item = obj
        if not recipient and obj:
            # First capitalized token in the object is the recipient.
            m = re.match(r"^([A-Z][a-zA-Z]*)\s+(?:the\s+)?(.+)$", obj)
            if m:
                recipient, item = _norm(m.group(1)), _norm(m.group(2))
        if not item or not recipient:
            return
        giver = self._entity(subjects[0])
        if _norm(item) in giver.carrying:
            giver.carrying.remove(_norm(item))
        self._take(recipient, item)
        self._log(subjects[0], "gave", f"{item} to {recipient}")

    def _log(self, subject: str, pred: str, obj: str) -> None:
        """Append a bound, event-indexed SituationEvent.

        Every logged clause becomes a structured event: participants
        bound (protagonist index), situated (space index), sequenced
        (time index), carrying stated cause and purpose (causation and
        intention indices). The event then binds to recent events that
        share a participant or location — the construction–integration
        step that turns a list of clauses into a coherent model. An
        event that binds to nothing marks a new episode.
        """
        ctx = self._ctx
        # An embedded clause carries its content in ctx.obj (the
        # object slot is empty — the object IS a proposition).
        if not obj and ctx is not None and ctx.extra_participants:
            obj = ctx.obj
        participants: list[str] = []
        for cand in [subject, *(ctx.subjects if ctx else ())]:
            if cand and cand not in participants:
                participants.append(cand)
        # Object parts count as participants unless they are the
        # event's own place ("moved to office" — the office is the
        # spatial index, not a participant in the moving) — and
        # unless the object is clause content, which is described,
        # not a referent.
        excluded = {ctx.location, ctx.destination} if ctx else set()
        embedded = bool(ctx and ctx.extra_participants)
        if not embedded:
            for part in re.split(r"\s+(?:to|and)\s+", obj):
                name = _norm(part)
                if name and name not in participants and name not in excluded:
                    participants.append(name)
        # Entities inside an embedded clause bind the attribution
        # event to the discourse — "John believes that Mary is in the
        # kitchen" is an event about John, Mary, and the kitchen.
        if ctx is not None:
            for cand in ctx.extra_participants:
                if cand and cand not in participants:
                    participants.append(cand)

        location = ""
        if ctx is not None:
            location = ctx.location or ctx.destination

        ev = SituationEvent(
            seq=self._seq,
            episode=self._episode,
            subject=subject,
            predicate=pred,
            object=obj,
            participants=tuple(participants),
            location=location,
            time_marker=ctx.time_marker if ctx else "",
            tense=ctx.tense if ctx else "unknown",
            negated=ctx.negated if ctx else False,
            uncertain=ctx.uncertain if ctx else False,
            cause_text=ctx.cause_text if ctx else "",
            intention=ctx.intention if ctx else "",
        )

        # Construction–integration: bind to recent events sharing a
        # participant or the location. An unbound event signals a
        # discontinuity — a new episode of the discourse.
        bound: list[int] = []
        ev_parts = set(ev.participants)
        for prev in self.events[-self.BIND_WINDOW :]:
            if prev.seq == ev.seq:
                continue
            if ev_parts & set(prev.participants) or (
                ev.location and ev.location == prev.location
            ):
                bound.append(prev.seq)
        if not bound and self.events:
            self._episode += 1
            ev.episode = self._episode
            self.episode_shifted = True
        ev.bound_to = tuple(bound)

        self.events.append(ev)
        for p in ev.participants:
            self.by_participant[p].append(ev.seq)
        if ev.location:
            self.by_location[ev.location].append(ev.seq)
        self.by_episode[ev.episode].append(ev.seq)
        self._seq += 1

    def _record_request(self, result: ComprehensionResult) -> None:
        """Record a question or command as an interlocutor intention.

        The addressee's speech acts are goals directed at Genesis —
        "tell me about X" is something the user wants her to do. They
        go on the intention index (requests), not the world state.
        """
        kind = "command" if result.is_command else "question"
        for prop in result.propositions:
            theme = prop.roles.get(_ROLE_THEME) or ""
            self.requests.append(
                DialogueAct(
                    seq=self._req_seq,
                    kind=kind,
                    subject=_norm(prop.subject),
                    predicate=prop.predicate.lower(),
                    object=_norm(prop.object),
                    focus=prop.focus or result.focus,
                    theme=_norm(theme),
                )
            )
            self._req_seq += 1
        del self.requests[: -self.MAX_REQUESTS]

    def _link_stated_causes(
        self, result: ComprehensionResult, start_seq: int
    ) -> None:
        """Wire explicit consequence markers between events.

        "A so B" and "A. Therefore B" state that A caused B; both
        split into separate clauses, so the events exist and only the
        causal edge needs adding. A sentence-initial "therefore" links
        the previous statement's last event to this one.
        """
        new = [e for e in self.events if e.seq >= start_seq]
        if not new:
            return
        raw_l = result.raw_text.lower()
        if re.match(r"^(?:therefore|thus|hence)\b", raw_l):
            prev = next(
                (e for e in reversed(self.events) if e.seq < start_seq),
                None,
            )
            if prev is not None:
                self._add_cause(prev.seq, new[0].seq)
        if len(new) >= 2 and re.search(r"\b(?:so|therefore|thus|hence)\b", raw_l):
            for effect in new[1:]:
                self._add_cause(new[0].seq, effect.seq)

    def _add_cause(self, cause_seq: int, effect_seq: int) -> None:
        """Record that one event caused/enable another."""
        if cause_seq == effect_seq:
            return
        if not (0 <= cause_seq < len(self.events)
                and 0 <= effect_seq < len(self.events)):
            return
        cause = self.events[cause_seq]
        effect = self.events[effect_seq]
        if effect_seq not in cause.causes:
            cause.causes = (*cause.causes, effect_seq)
        if cause_seq not in effect.caused_by:
            effect.caused_by = (*effect.caused_by, cause_seq)

    # ─── Queries ─────────────────────────────────────────────────

    def where_is(self, entity: str) -> tuple[str | None, set[str]]:
        """(definite location, uncertain candidates) for an entity."""
        e = self.entities.get(_norm(entity))
        if e is None:
            return None, set()
        return e.location, set(e.candidates)

    def is_at(self, entity: str, place: str) -> str | None:
        """"yes" / "no" / "maybe" — or None when nothing is known."""
        e = self.entities.get(_norm(entity))
        if e is None:
            return None
        p = _place(place)
        if e.location is not None:
            return "yes" if e.location == p else "no"
        if p in e.candidates:
            return "maybe"
        if e.candidates:
            return "no"
        if p in e.not_at:
            return "no"
        return None

    def who_is_at(self, place: str) -> list[str]:
        p = _place(place)
        return sorted(
            name for name, e in self.entities.items()
            if e.location == p and not e.holder
        )

    def carrying(self, entity: str) -> list[str]:
        e = self.entities.get(_norm(entity))
        return list(e.carrying) if e else []

    def count_carried(self, entity: str) -> int | None:
        e = self.entities.get(_norm(entity))
        return len(e.carrying) if e else None

    def where_was_before(self, entity: str, place: str) -> str | None:
        """The entity's location immediately before arriving at place."""
        e = self.entities.get(_norm(entity))
        if e is None:
            return None
        p = _place(place)
        prev: str | None = None
        for _, loc in e.location_history:
            if loc == p:
                return prev
            prev = loc
        return None

    def who_gave(self, item: str, recipient: str) -> str | None:
        """Who gave <item> to <recipient> — scans the event log."""
        want_item = _norm(item)
        want_rcpt = _norm(recipient)
        for ev in reversed(self.events):
            if ev.predicate != "gave":
                continue
            m = re.match(r"^(.+?)\s+to\s+(.+)$", ev.object)
            if not m:
                continue
            if _norm(m.group(1)) == want_item and _norm(m.group(2)) == want_rcpt:
                return ev.subject
        return None

    def last_action_object(self, entity: str) -> tuple[str, str] | None:
        """Most recent (predicate, object) event for an entity."""
        name = _norm(entity)
        for ev in reversed(self.events):
            if ev.subject == name and ev.object and ev.predicate not in ("moved to", "is"):
                return ev.predicate, ev.object
        return None

    # ─── Event-index queries ─────────────────────────────────────
    # The indices that make the log a model rather than a transcript:
    # protagonist, space, time, causation, intention (Zwaan &
    # Radvansky, 1998).

    def events_involving(
        self, entity: str, limit: int = 10
    ) -> list[SituationEvent]:
        """Events an entity participates in, most recent first —
        the protagonist index."""
        seqs = self.by_participant.get(_norm(entity), [])
        return [self.events[s] for s in reversed(seqs[-limit:])]

    def recent_events(self, n: int = 4) -> list[SituationEvent]:
        """The n most recent events — the recency edge of the model."""
        return self.events[-n:]

    def why_did(self, entity: str, predicate: str = "") -> str | None:
        """The stated cause of an entity's most recent matching event.

        Answers "why did X <predicate>?" from the causation index:
        an explicit "because" clause wins, then a linked cause event
        ("A so B"), else None — the model does not invent reasons.
        Predicate matching is morphological ("go" matches "went") and
        family-aware ("go" matches the logged transition "moved to").
        """
        name = _norm(entity)
        for ev in reversed(self.events):
            if name not in ev.participants:
                continue
            if predicate and not _verb_match(predicate, ev.predicate):
                continue
            if ev.cause_text:
                return ev.cause_text
            if ev.caused_by:
                cause = self.events[ev.caused_by[-1]]
                return cause.describe()
            return None
        return None

    def intentions_of(self, entity: str) -> list[str]:
        """An entity's stated purposes — the intention index.

        "John went to the store to buy milk" stores "buy milk" as
        John's intention. Only purposes stated in the discourse are
        returned; the model does not infer goals.
        """
        name = _norm(entity)
        out: list[str] = []
        for ev in self.events:
            if name in ev.participants and ev.intention:
                if ev.intention not in out:
                    out.append(ev.intention)
        return out

    def beliefs_of(self, entity: str) -> list[str]:
        """Content attributed to an entity's mind — "what does X
        believe/think/hope?" answered from attribution, not fact.

        These are propositions the discourse *ascribed* to the entity
        ("John believes that Mary is in the kitchen"), stored under
        John's state without asserting them. Most recent last, capped
        at ``_MAX_BELIEFS``.
        """
        ent = self.entities.get(_norm(entity))
        return list(ent.beliefs) if ent else []

    def reports_of(self, entity: str) -> list[str]:
        """Content an entity publicly said/reported — speech acts, not
        mental states. Searched on the event log: any event whose
        subject is the entity and whose predicate is a report verb
        carries its content in ``object``.
        """
        name = _norm(entity)
        out: list[str] = []
        for ev in self.events:
            if ev.subject != name or not ev.object:
                continue
            base = is_verb_form(ev.predicate) or ev.predicate
            if base in _CTP_REPORT:
                out.append(ev.object)
        return out

    def active_participants(self, limit: int = 6) -> list[str]:
        """Entities bound in the current episode, most recent first.

        These are the referents held "in mind" — what working memory
        should keep active for pronoun resolution and topic continuity
        (Baddeley's episodic buffer function: the current bound
        structure, not loose items).
        """
        out: list[str] = []
        for ev in reversed(self.events):
            if ev.episode != self._episode:
                break
            for p in ev.participants:
                if p not in out:
                    out.append(p)
                    if len(out) >= limit:
                        return out
        return out

    def recent_relations(self, n: int = 4) -> list[tuple[str, str, str]]:
        """Recent events as (subject, predicate, object) triples —
        the relational structure working memory binds."""
        out: list[tuple[str, str, str]] = []
        for ev in self.events[-n:]:
            target = ev.object or ev.location
            if ev.subject and target:
                out.append((ev.subject, ev.predicate, target))
        return out

    def open_requests(self) -> list[DialogueAct]:
        """The interlocutor's pending questions and commands."""
        return list(self.requests)

    @property
    def current_episode(self) -> int:
        """The episode the most recent event belongs to."""
        return self._episode

    @property
    def episode_count(self) -> int:
        """How many episodes the discourse has segmented into."""
        return self._episode + 1 if self.events else 0

    def consume_episode_shift(self) -> bool:
        """Read-and-clear the episode-boundary flag."""
        shifted = self.episode_shifted
        self.episode_shifted = False
        return shifted

    def describe(self) -> str:
        """A one-line structural summary for introspection."""
        if not self.events:
            return "Situation model: nothing in mind."
        parts = ", ".join(self.active_participants()) or "no one"
        last = self.events[-1].describe()
        return (
            f"Situation model: episode {self._episode} of "
            f"{self.episode_count}, {len(self.events)} events, "
            f"active: {parts}. Last: {last}."
        )

    def direction_of(self, place: str, rel: str) -> str | None:
        """What is <rel> of <place>: 'north of the hallway' → kitchen."""
        p = _place(place)
        for src, dirs in self.place_dir.items():
            if dirs.get(rel) == p:
                return src
            # Converse: "a is d of place" means place's opposite(d) is a,
            # so if the query asks opposite(d) of a, the answer is b —
            # i.e. a is north of hallway ⇒ hallway is south of garden
            # when a="garden", d="north", b="hallway".
            for d, b in dirs.items():
                if src == p and _OPPOSITE.get(d) == rel:
                    return b
        return None

    def place_relation(self, a: str, rel: str, b: str) -> str | None:
        """Is a <rel> of b?  (positional: left/right/above/below)."""
        rels = self.pos_rel.get(_norm(a), {})
        b_name = _norm(b)
        if rels.get(rel) == b_name:
            return "yes"
        # Known to stand in a different relation to b, or under the
        # converse — e.g. circle is right of square ⇒ not left of it.
        if rels.get(_POS_OPPOSITE.get(rel, "")) == b_name:
            return "no"
        if b_name in rels.values():
            return "no"
        return None

    def is_bigger(self, a: str, b: str) -> str | None:
        """Transitive size query: is a bigger than b?"""
        bigger = self._bigger_than(b)
        if _norm(a) in bigger:
            return "yes"
        if _norm(b) in self._bigger_than(a):
            return "no"
        return None

    def fits_in(self, a: str, b: str) -> str | None:
        """Does a fit in b?  a fits in b iff a is smaller than b."""
        if _norm(b) in self._bigger_than(a):
            return "yes"
        if _norm(a) in self._bigger_than(b):
            return "no"
        return None

    def _bigger_than(self, x: str) -> set[str]:
        """Everything transitively bigger than x (BFS over smaller_than)."""
        out: set[str] = set()
        queue = deque(self.smaller_than.get(_norm(x), ()))
        while queue:
            n = queue.popleft()
            if n in out:
                continue
            out.add(n)
            queue.extend(self.smaller_than.get(n, ()))
        return out

    def path(self, src: str, dst: str) -> str | None:
        """Direction path src→dst over place_dir ('n,s'), or None."""
        start, goal = _norm(src), _norm(dst)
        # place_dir[a][d]=b means "a is d of b": travelling from a to b
        # moves opposite(d), and from b to a moves d.
        graph: dict[str, dict[str, str]] = defaultdict(dict)
        for a, dirs in self.place_dir.items():
            for d, b in dirs.items():
                graph[a][_OPPOSITE[d]] = b
                graph[b][d] = a
        if start == goal:
            return ""
        seen = {start}
        queue: deque[tuple[str, list[str]]] = deque([(start, [])])
        while queue:
            node, path = queue.popleft()
            for d, nxt in graph.get(node, {}).items():
                if nxt in seen:
                    continue
                new = [*path, d]
                if nxt == goal:
                    return ",".join(_DIR_LETTER[x] for x in new)
                seen.add(nxt)
                queue.append((nxt, new))
        return None

    def state_of(self, entity: str) -> str | None:
        """Most recent state attribute (hungry, tired, ...) — for
        'why did X go' motivation questions."""
        e = self.entities.get(_norm(entity))
        if e is None:
            return None
        states = e.attributes & _STATE_WORDS
        return sorted(states)[0] if states else None

    def fears_of(self, entity: str) -> list[str]:
        """What the entity is afraid of — own fears plus fears of its
        classes (deduction: 'mice are afraid of cats', 'Jessica is a
        mouse' → Jessica is afraid of cats)."""
        e = self.entities.get(_norm(entity))
        if e is None:
            return []
        feared = set(e.fears)
        for cls in e.classes:
            feared |= self.entities.get(cls, EntityState()).fears
        return sorted(feared)

    def color_of(self, entity: str) -> str | None:
        """Entity's color — own attribute, else induced from a class
        member's color (induction: 'Lily is a swan and white; Greg is a
        swan' → Greg is white)."""
        e = self.entities.get(_norm(entity))
        if e is None:
            return None
        own = e.attributes & _COLORS
        if own:
            return sorted(own)[0]
        for cls in e.classes:
            for name, other in self.entities.items():
                if cls in other.classes and name != _norm(entity):
                    induced = other.attributes & _COLORS
                    if induced:
                        return sorted(induced)[0]
        return None


_OPPOSITE = {"north": "south", "south": "north",
             "east": "west", "west": "east"}
_POS_OPPOSITE = {"left": "right", "right": "left",
                 "above": "below", "below": "above"}
_DIR_LETTER = {"north": "n", "south": "s", "east": "e", "west": "w"}

# SemanticRole members resolved lazily to avoid a hard import cycle at
# module load (comprehension imports nothing from cognition).
from ..language.comprehension import SemanticRole as _SR  # noqa: E402

_ROLE_GOAL = _SR.GOAL
_ROLE_LOCATION = _SR.LOCATION
_ROLE_ATTRIBUTE = _SR.ATTRIBUTE
_ROLE_BENEFICIARY = _SR.BENEFICIARY
_ROLE_TIME = _SR.TIME
_ROLE_THEME = _SR.THEME
