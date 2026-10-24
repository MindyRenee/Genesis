"""Correspondence — truth as a relation between a proposition and a state of affairs.

The correspondence theory of truth (Russell 1912; Austin 1950; Tarski's
Convention T) says a truth-bearer is true iff the state of affairs it
denotes obtains. In this architecture the truth-bearer is a
``Proposition`` — comprehension's "smallest unit of meaning that can
be true or false" — and the state of affairs is the parietal state:
what the situation model has settled from discourse.

Two of Austin's conventions map onto the implementation directly:

- **Demonstrative conventions** pick *which* situation a statement
  refers to — here, entity resolution: a proposition's subject and
  object normalize through the same ``_norm``/``_place`` the model
  applies, so a claim binds the same entity the model stores.
- **Descriptive conventions** pick which *type* of situation the
  sentence describes — here, the claim kind: location, possession,
  attribute, positional relation, direction, size order, fear, or
  attributed belief. Each maps to the predicate the model actually
  stores, never to a substring of the sentence.

Russell's congruence is the tuple check: claimed (S, R, O) against
the modeled fact. A state of affairs "obtains" when the model's
settled state structurally matches the claim — location equals,
holder equals, attribute present, relation present.

Truth under partial observation is graded, not bivalent:

- Each check yields κ ∈ {+1, −1, ±w, ∅}: the model affirms the state
  (SUPPORTED), affirms its negation (CONTRADICTED), bears on it
  equivocally (UNSETTLED — "maybe"), or has no bearing (UNKNOWN).
- κ folds into a Beta posterior τ(p) = α/(α+β) — the correspondence
  *degree*, with ``evidence`` counting the observations behind it.
  The Beta(1,1) prior is honest: a proposition never checked sits at
  degree 0.5 with zero evidence — present, but unasserted.
- An UNSETTLED check counts a half-weight observation on *both*
  sides: the model spoke, but equivocally — evidence accumulates
  while the mean holds mid, so a contested claim reads as contested
  rather than as unsupported.
- Convention T is the limit, not the mechanism: as confirmations
  accumulate without contradiction, τ → 1 — "p is true" approaches
  "p obtains". The ledger reports posterior mass; it never asserts
  truth itself.

Attribution:

    "John believes that Mary is in the kitchen" claims something
    about John's model, not the world — its state of affairs is
    membership in John's stated belief set (``beliefs_of``), not
    Mary's location. Checking the embedded content against the world
    would collapse "what John thinks" into "what is" — the scope
    error the situation model's belief index exists to prevent.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum, auto
from typing import TYPE_CHECKING

from parietal_lobe.belief import Beta
from parietal_lobe.situation_model import (
    _AFRAID_RE,
    _COPULA,
    _DIR_OF_RE,
    _DROP,
    _GERUND_POSSESSION_RE,
    _GIVE,
    _MOTION,
    _OPPOSITE,
    _PARTICIPLE_LEMMA,
    _POS_OF_RE,
    _TAKE,
    _describe_prop,
    _norm,
    _place,
)
from temporal_lobe.comprehension import SemanticRole
from temporal_lobe.morphology import deconjugate_verb

if TYPE_CHECKING:
    from parietal_lobe.situation_model import SituationModel
    from temporal_lobe.comprehension import Proposition

#: Upper bound on tracked proposition posteriors — evicts the
#: least-evidenced claim when full (same convention as the belief
#: layer's bounded topic map).
_MAX_PROPOSITIONS = 256

#: An equivocal check ("maybe") counts this much weight on each side:
#: the evidence grows — the model did speak — while the degree holds.
_UNSETTLED_W = 0.5

# Claim kinds — the descriptive half of correspondence: which *type*
# of situation the sentence describes, mapped to the predicate the
# model stores.
_LOCATED = "located"
_CARRIES = "carries"
_ATTRIBUTE = "attribute"
_POS_REL = "pos_rel"
_DIR = "direction"
_SIZE_GT = "size_gt"
_FEARS = "fears"
_BELIEVES = "believes"

# Predicates that assert a state of affairs the model does not
# represent (event-only claims) yield no claim at all — the model
# has no bearing, and UNKNOWN is the honest verdict.
_HAVE = frozenset(
    {"have", "has", "had", "own", "owns", "owned", "possess", "possesses", "possessed"}
)

# "X is bigger than the box" — the size claim rides in the copula's
# object/attribute, not the predicate.
_SIZE_ATTR_RE = re.compile(r"^\s*(bigger|larger|smaller)\s+than\s+(?:the\s+)?(.+?)\s*$", re.I)


class Verdict(Enum):
    """The model's bearing on a state of affairs — one check's κ."""

    SUPPORTED = auto()  # +1 — the model affirms the state
    CONTRADICTED = auto()  # −1 — the model affirms its negation
    UNSETTLED = auto()  # ±w — the model bears, equivocally
    UNKNOWN = auto()  # ∅ — the model has no bearing


# The package surface already exports ``Verdict`` for the symbolic
# math layer — the truth check goes out under its own name.
TruthVerdict = Verdict


@dataclass(frozen=True)
class Claim:
    """The state of affairs a proposition refers to.

    Austin's demonstrative half, realized: normalized relata plus the
    descriptive kind, so the same sentence always binds the same
    state — and so the ledger accumulates evidence for the *claim*,
    not the string that carried it.
    """

    kind: str
    subject: str
    target: str = ""
    rel: str = ""
    negated: bool = False

    def key(self) -> tuple:
        """Canonical ledger key — evidence for a claim pools across
        the paraphrases that assert it."""
        return (self.kind, self.subject, self.target, self.rel, self.negated)


@dataclass(frozen=True)
class Correspondence:
    """One evaluation of a proposition against the modeled world.

    ``verdict`` is this check's bearing; ``degree`` and ``evidence``
    are the running posterior over the claim's correspondence —
    τ(p) and how much observation stands behind it. ``detail`` names
    the model state that bore on the check, so a verdict can be
    traced to the fact that made it.
    """

    claim: Claim | None
    verdict: Verdict
    degree: float
    evidence: float
    detail: str = ""

    @property
    def true(self) -> bool | None:
        """The bivalent shadow Convention T licenses: True on
        SUPPORTED, False on CONTRADICTED, None while the model is
        equivocal or silent. The graded picture lives in ``degree``."""
        if self.verdict is Verdict.SUPPORTED:
            return True
        if self.verdict is Verdict.CONTRADICTED:
            return False
        return None


def _claim_of(prop: Proposition) -> Claim | None:
    """Map a proposition to the state of affairs it refers to.

    Returns None for truth-bearers the model has no bearing on —
    event claims with no settled state residue. The verb families
    and normalizers are the situation model's own: a claim means
    exactly what the model would have stored had it been asserted.
    """
    if prop.focus or not prop.subject:
        return None
    subject = _norm(prop.subject)
    if prop.embedded is not None:
        # Attributed content: the claim is about the holder's belief
        # set — "john believes <content>" refers to a state of his
        # mind, rendered exactly as the model stores it.
        return Claim(
            _BELIEVES,
            subject,
            _describe_prop(prop.embedded),
            negated=prop.negated,
        )
    # Passive voice names the patient as subject; the agent role
    # holds who acted — "the ball was taken by Mary" claims about
    # Mary's hands, not the ball's.
    if prop.passive:
        agent = prop.roles.get(SemanticRole.AGENT)
        if agent:
            subject = _norm(agent)
    pred = (prop.predicate or "").strip().lower()
    # deconjugate_verb returns candidate lemmas (["took", "take"]) —
    # the predicate belongs to a family if any candidate does.
    bases = {pred} | set(deconjugate_verb(pred))
    negated = prop.negated or pred.endswith("n't") or any(b.endswith("n't") for b in bases)

    if bases & _COPULA:
        return _copula_claim(prop, subject, negated)

    if bases & _MOTION:
        # Destinations arrive as GOAL ("went to the kitchen") or
        # LOCATION ("ran in the park") depending on the parse.
        dest = (
            prop.roles.get(SemanticRole.LOCATION)
            or prop.roles.get(SemanticRole.GOAL)
            or prop.object
        )
        if dest:
            return Claim(_LOCATED, subject, _place(dest), negated=negated)
        return None

    if bases & (_TAKE | _HAVE):
        if prop.object:
            return Claim(_CARRIES, subject, _norm(prop.object), negated=negated)
        return None

    if bases & _DROP:
        if prop.object:
            return Claim(_CARRIES, subject, _norm(prop.object), negated=True)
        return None

    if bases & _GIVE:
        # The recipient rides GOAL in a "gave X to Y" parse.
        beneficiary = prop.roles.get(SemanticRole.BENEFICIARY) or prop.roles.get(SemanticRole.GOAL)
        if beneficiary and prop.object:
            return Claim(_CARRIES, _norm(beneficiary), _norm(prop.object), negated=negated)
        return None

    return None


def _copula_claim(prop: Proposition, subject: str, negated: bool) -> Claim | None:
    """Map a copular proposition to the state of affairs it refers to.

    Copula syntax hides several claim shapes — location ("is in the
    park"), possession wearing "is carrying" clothes, fear ("is afraid
    of"), size comparisons, and spatial attributes — each rendered as
    the situation model would have stored it.
    """
    loc = prop.roles.get(SemanticRole.LOCATION)
    if loc:
        return Claim(_LOCATED, subject, _place(loc), negated=negated)
    attr = prop.roles.get(SemanticRole.ATTRIBUTE, "")
    for text in (prop.object or "", attr):
        if not text:
            continue
        # "is carrying the milk" — a possession wearing copula
        # syntax; the participle maps to its verb family lemma.
        m = _GERUND_POSSESSION_RE.match(text.strip())
        if m:
            lemma = _PARTICIPLE_LEMMA.get(m.group(1).lower(), m.group(1).lower())
            item = _norm(m.group(2))
            if lemma in _DROP:
                # "is dropping the ball" claims her hands end up
                # empty — a negated carries state.
                return Claim(_CARRIES, subject, item, negated=True)
            return Claim(_CARRIES, subject, item, negated=negated)
        m = _AFRAID_RE.search(text)
        if m:
            return Claim(_FEARS, subject, _norm(m.group(1)), negated=negated)
        m = _SIZE_ATTR_RE.match(text)
        if m:
            word, other = m.group(1).lower(), _norm(m.group(2))
            if word == "smaller":
                # "a is smaller than b" ≡ "b is bigger than a" —
                # the same state of affairs inverts the relata.
                return Claim(_SIZE_GT, other, subject, negated=negated)
            return Claim(_SIZE_GT, subject, other, negated=negated)
        m = _POS_OF_RE.search(text)
        if m:
            rel = (m.group(1) or m.group(3) or "").lower()
            target = _norm(m.group(2) or m.group(4) or "")
            if rel and target:
                return Claim(_POS_REL, subject, target, rel=rel, negated=negated)
        m = _DIR_OF_RE.search(text)
        if m:
            return Claim(
                _DIR,
                subject,
                _norm(m.group(2)),
                rel=m.group(1).lower(),
                negated=negated,
            )
    obj = prop.object or attr
    # A negated copula can leave the bare negator as the object
    # ("mary is not" → obj "not") — the negation is already on
    # the claim; a "not" attribute would claim nonsense.
    if obj and _norm(obj) not in ("not", "n't", "never"):
        return Claim(_ATTRIBUTE, subject, _norm(obj), negated=negated)
    return None


class CorrespondenceLedger:
    """Beta posteriors over proposition↔world correspondence.

    Bound to one situation model — the discourse-described world the
    claims are checked against. Each ``check`` evaluates a
    proposition's claim against the model's settled state and folds
    the bearing into the claim's running posterior, so evidence
    accumulates per *claim* across the paraphrases that assert it.
    """

    def __init__(self, model: SituationModel, max_propositions: int = _MAX_PROPOSITIONS) -> None:
        self._model = model
        self._max = max(1, max_propositions)
        self._posteriors: dict[tuple, Beta] = {}

    # ── Evaluation ──────────────────────────────────────────────

    def assess(self, prop: Proposition) -> Correspondence:
        """Evaluate a proposition without folding the result in —
        a read-only check that reports the running degree."""
        claim = _claim_of(prop)
        if claim is None:
            return Correspondence(None, Verdict.UNKNOWN, 0.5, 0.0)
        verdict, detail = self._bearing(claim)
        post = self._posteriors.get(claim.key())
        degree = post.mean if post is not None else 0.5
        evidence = post.evidence if post is not None else 0.0
        return Correspondence(claim, verdict, degree, evidence, detail)

    def check(self, prop: Proposition) -> Correspondence:
        """Evaluate a proposition and fold its bearing into the
        claim's correspondence posterior."""
        claim = _claim_of(prop)
        if claim is None:
            return Correspondence(None, Verdict.UNKNOWN, 0.5, 0.0)
        verdict, detail = self._bearing(claim)
        post = self._posterior(claim)
        self._fold(verdict, post)
        return Correspondence(claim, verdict, post.mean, post.evidence, detail)

    def degree_of(self, prop: Proposition) -> float | None:
        """The recorded correspondence degree for a proposition's
        claim — None when no check has ever borne on it."""
        claim = _claim_of(prop)
        if claim is None:
            return None
        post = self._posteriors.get(claim.key())
        return post.mean if post is not None else None

    # ── The model's bearing ─────────────────────────────────────

    def _bearing(self, claim: Claim) -> tuple[Verdict, str]:
        """Check the claim's state of affairs against settled state.

        The verdict lattice: does the model affirm the state, affirm
        its negation, bear equivocally, or stay silent? Negation then
        flips the decisive verdicts — the correspondence of ¬p is the
        non-obtaining of p.
        """
        kind = claim.kind
        verdict = Verdict.UNKNOWN
        detail = ""
        e = self._model.entities.get(claim.subject)

        if kind == _LOCATED:
            ans = self._model.is_at(claim.subject, claim.target)
            if ans == "yes":
                verdict, detail = Verdict.SUPPORTED, f"{claim.subject} is at {claim.target}"
            elif ans == "no":
                verdict, detail = Verdict.CONTRADICTED, f"{claim.subject} is not at {claim.target}"
            elif ans == "maybe":
                verdict, detail = Verdict.UNSETTLED, f"{claim.subject} might be at {claim.target}"
        elif kind == _CARRIES:
            verdict, detail = self._carries_bearing(claim, e)
        elif kind == _ATTRIBUTE:
            if e is not None and claim.target in (e.attributes | e.classes):
                verdict, detail = Verdict.SUPPORTED, f"{claim.subject} is {claim.target}"
        elif kind == _FEARS:
            if e is not None and claim.target in e.fears:
                verdict, detail = Verdict.SUPPORTED, f"{claim.subject} fears {claim.target}"
        elif kind == _POS_REL:
            ans = self._model.place_relation(claim.subject, claim.rel, claim.target)
            if ans == "yes":
                verdict, detail = (
                    Verdict.SUPPORTED,
                    f"{claim.subject} is {claim.rel} of {claim.target}",
                )
            elif ans == "no":
                verdict, detail = (
                    Verdict.CONTRADICTED,
                    f"{claim.subject} is not {claim.rel} of {claim.target}",
                )
        elif kind == _DIR:
            verdict, detail = self._dir_bearing(claim)
        elif kind == _SIZE_GT:
            ans = self._model.is_bigger(claim.subject, claim.target)
            if ans == "yes":
                verdict, detail = (
                    Verdict.SUPPORTED,
                    f"{claim.subject} is bigger than {claim.target}",
                )
            elif ans == "no":
                verdict, detail = (
                    Verdict.CONTRADICTED,
                    f"{claim.subject} is not bigger than {claim.target}",
                )
        elif kind == _BELIEVES:
            held = {_norm(b) for b in self._model.beliefs_of(claim.subject)}
            if _norm(claim.target) in held:
                verdict, detail = (
                    Verdict.SUPPORTED,
                    f"{claim.subject} believes {claim.target}",
                )

        if claim.negated:
            if verdict is Verdict.SUPPORTED:
                verdict = Verdict.CONTRADICTED
            elif verdict is Verdict.CONTRADICTED:
                verdict = Verdict.SUPPORTED
        return verdict, detail

    def _carries_bearing(self, claim: Claim, entity) -> tuple[Verdict, str]:
        """Possession congruence: the holder's list AND the object's
        backlink. An object recorded under another holder is a real
        contradiction — two people cannot carry the same thing."""
        if entity is not None and claim.target in entity.carrying:
            return Verdict.SUPPORTED, f"{claim.subject} carries {claim.target}"
        other = self._model.entities.get(claim.target)
        if other is None:
            return Verdict.UNKNOWN, ""
        if other.holder == claim.subject:
            return Verdict.SUPPORTED, f"{claim.subject} carries {claim.target}"
        if other.holder:
            return Verdict.CONTRADICTED, f"{claim.target} is held by {other.holder}"
        # A tracked object with an empty holder is a free object —
        # the model affirming "held by no one" contradicts the claim.
        return Verdict.CONTRADICTED, f"{claim.target} is held by no one"

    def _dir_bearing(self, claim: Claim) -> tuple[Verdict, str]:
        """Direction claims — 'the kitchen is north of the hallway' —
        against the model's place-direction graph, converse included."""
        dirs = self._model.place_dir.get(claim.subject, {})
        actual = dirs.get(claim.rel)
        if actual == claim.target:
            return Verdict.SUPPORTED, f"{claim.subject} is {claim.rel} of {claim.target}"
        if actual is not None:
            return (
                Verdict.CONTRADICTED,
                f"{claim.subject} is {claim.rel} of {actual}, not {claim.target}",
            )
        # Converse bearing: "kitchen is north of hallway" is also
        # contradicted when hallway is recorded north of kitchen.
        converse = _OPPOSITE.get(claim.rel, "")
        back = self._model.place_dir.get(claim.target, {})
        if converse and back.get(converse) == claim.subject:
            return Verdict.SUPPORTED, f"{claim.subject} is {claim.rel} of {claim.target}"
        if back.get(claim.rel) == claim.subject:
            return Verdict.CONTRADICTED, f"{claim.target} is {claim.rel} of {claim.subject}"
        return Verdict.UNKNOWN, ""

    # ── Posterior bookkeeping ───────────────────────────────────

    @staticmethod
    def _fold(verdict: Verdict, post: Beta) -> None:
        """Fold one check's κ into the posterior. UNKNOWN contributes
        nothing — the model's silence is not evidence."""
        if verdict is Verdict.SUPPORTED:
            post.observe(True)
        elif verdict is Verdict.CONTRADICTED:
            post.observe(False)
        elif verdict is Verdict.UNSETTLED:
            post.observe(True, _UNSETTLED_W)
            post.observe(False, _UNSETTLED_W)

    def _posterior(self, claim: Claim) -> Beta:
        """Get or create the claim's posterior, keeping the ledger
        bounded — the least-evidenced claim evicts first."""
        post = self._posteriors.get(claim.key())
        if post is not None:
            return post
        if len(self._posteriors) >= self._max:
            weakest = min(self._posteriors, key=lambda k: self._posteriors[k].evidence)
            del self._posteriors[weakest]
        post = Beta()
        self._posteriors[claim.key()] = post
        return post

    @property
    def tracked(self) -> int:
        """How many claims carry a posterior."""
        return len(self._posteriors)

    def describe(self) -> str:
        """A one-line structural summary for introspection."""
        if not self._posteriors:
            return "Correspondence ledger: nothing checked."
        most = max(self._posteriors.values(), key=lambda p: p.evidence)
        return (
            f"Correspondence ledger: {len(self._posteriors)} claims, "
            f"deepest evidence {most.evidence:.1f} at degree {most.mean:.2f}."
        )
