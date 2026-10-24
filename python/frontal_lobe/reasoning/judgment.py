"""Judgment — the act of taking a stance on a proposition.

Frege's judgment stroke: to entertain a thought is one thing, to
*judge* it is another — judgment is acknowledging the content as
true (or refusing to), the step from holding a representation to
committing on it. Kant's table makes the same point structurally:
a judgment is a determinate relation of a content to truth.

In this architecture the content already exists — a ``Proposition``
from comprehension, a ``Correspondence`` check from the parietal
ledger, a ``CriticalAssessment`` from epistemic scrutiny. What was
missing is the commitment: a verdict is what the world bears, a
judgment is what *she* does with it — assert, hedge, suspend, or
deny. The stance is then the datum speech posture is composed from:
an asserted claim is said plainly, a hedged one qualified, a denied
one answered with its negation, a suspended one honestly deferred.

The mapping is not arbitrary — it falls out of the evidence:

- SUPPORTED with posterior τ ≥ ``_ASSERT_T`` → ASSERT. The Beta
  accounting makes this roughly "two or more net supports"; the
  first confirming check alone never licenses assertion.
- CONTRADICTED with τ ≤ ``_DENY_T`` → DENY: confident enough in the
  negation to say so. A contradiction on thin evidence still hedges.
- Anything else → SUSPEND or HEDGE. UNSETTLED bearing (the model
  said "either/or") is a real suspension of judgment, not a low
  confidence score; UNKNOWN is the same with no bearing at all.

Reasoning-derived claims come through ``of_assessment``: the
critical thinker's ACCEPT / HEDGE / REJECT / INVESTIGATE
recommendation is already a proto-stance; judgment renders it in
the same vocabulary so downstream speech needs one interface.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto

from parietal_lobe.correspondence import Correspondence, CorrespondenceLedger, Verdict
from parietal_lobe.situation_model import _describe_prop
from temporal_lobe.comprehension import Proposition

from .critical_thinking import AssessmentRecommendation, CriticalAssessment

# Stance thresholds on the claim's truth-degree posterior τ. ASSERT
# needs ~2 net supporting checks (Beta(1,1): τ = (1+s)/(2+s+c) ≥ 0.7
# at s−c ≥ 2.3); DENY mirrors it for confident negation.
_ASSERT_T = 0.70
_DENY_T = 0.30


class Stance(Enum):
    """What her judgment does with a content — the assertoric force."""

    ASSERT = auto()  # the claim obtains — say it
    HEDGE = auto()  # leaning, but the evidence is thin — qualify it
    SUSPEND = auto()  # equivocal bearing or none — hold judgment
    DENY = auto()  # the claim fails — assert its negation


@dataclass(frozen=True, slots=True)
class Judgment:
    """One adjudicated content: stance, degree, and what it rests on.

    ``content`` is the judged claim rendered as text; ``degree`` the
    truth-degree posterior behind the stance; ``evidence`` how much
    checking stands behind it; ``grounds`` the model states or
    assessment findings the judgment cites.
    """

    content: str
    stance: Stance
    degree: float
    evidence: float
    grounds: tuple[str, ...] = ()

    @property
    def committed(self) -> bool:
        """Whether a stance was taken — assert or deny, not deferred."""
        return self.stance in (Stance.ASSERT, Stance.DENY)

    @property
    def affirms(self) -> bool | None:
        """The bivalent shadow: True asserts, False denies, None defers."""
        if self.stance is Stance.ASSERT:
            return True
        if self.stance is Stance.DENY:
            return False
        return None


class Judge:
    """The assertion gate: propositions and assessments in, stances out.

    Holds the correspondence ledger she judges world-claims against;
    reasoning-derived claims arrive already assessed and go through
    :meth:`of_assessment` — the same stance vocabulary either way.
    """

    def __init__(self, ledger: CorrespondenceLedger | None = None) -> None:
        """Bind the ledger judgments are checked through, or None to
        judge without a world (everything suspends)."""
        self.ledger = ledger

    def judge(self, prop: Proposition) -> Judgment:
        """Judge a parsed proposition against the modeled world.

        A proposition without a claim — a verb-less fragment the
        model has no bearing on — suspends, which is the honest
        judgment: judgment presupposes a checkable content.
        """
        if self.ledger is None:
            return Judgment(_describe_prop(prop), Stance.SUSPEND, 0.5, 0.0)
        return self.of(self.ledger.check(prop), content=_describe_prop(prop))

    def of(self, corr: Correspondence, content: str = "") -> Judgment:
        """Render a stance from one correspondence check."""
        if corr.claim is not None and not content:
            content = " ".join(
                p for p in (corr.claim.subject, corr.claim.rel, corr.claim.target) if p
            )
        grounds = (corr.detail,) if corr.detail else ()
        return Judgment(
            content,
            _stance_of(corr.verdict, corr.degree),
            corr.degree,
            corr.evidence,
            grounds,
        )

    def of_assessment(self, content: str, assessment: CriticalAssessment) -> Judgment:
        """Render a stance from a critical assessment of reasoning.

        ACCEPT asserts, HEDGE hedges, REJECT denies, INVESTIGATE
        suspends — the epistemic recommendation mapped into the same
        assertoric vocabulary the world-checked claims use.
        """
        stance = _ASSESSMENT_STANCE[assessment.recommendation]
        grounds = tuple(assessment.fallacies) + tuple(assessment.notes[:2])
        evidence = len(assessment.supporting_evidence) + len(
            assessment.disconfirming_evidence
        )
        return Judgment(
            content,
            stance,
            assessment.revised_confidence,
            float(evidence),
            grounds,
        )


def _stance_of(verdict: Verdict, degree: float) -> Stance:
    """Verdict × posterior → assertoric force."""
    if verdict is Verdict.SUPPORTED:
        return Stance.ASSERT if degree >= _ASSERT_T else Stance.HEDGE
    if verdict is Verdict.CONTRADICTED:
        return Stance.DENY if degree <= _DENY_T else Stance.HEDGE
    return Stance.SUSPEND


_ASSESSMENT_STANCE = {
    AssessmentRecommendation.ACCEPT: Stance.ASSERT,
    AssessmentRecommendation.HEDGE: Stance.HEDGE,
    AssessmentRecommendation.REJECT: Stance.DENY,
    AssessmentRecommendation.INVESTIGATE: Stance.SUSPEND,
}
