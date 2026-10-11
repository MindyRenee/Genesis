"""Hypothesis-layer tests — falsifiable mechanics and inferred goals.

The machinery lives in ``reasoning/hypotheses.py`` and rides on
``TaskCompetence``: transitions are trials for mechanic rules,
terminal states are abductive evidence for goal conditions, and a
satisfied goal that play moves past is falsified.
"""

from __future__ import annotations

from frontal_lobe.reasoning.competence import GoalCondition, TaskCompetence
from frontal_lobe.reasoning.hypotheses import Hypothesis, HypothesisLedger, transition_view


def _ctx(competence: TaskCompetence, state: dict):
    return competence.recognize(
        domain="test.env",
        state=state,
        actions=("1", "2"),
        roles=("self",),
    )


# ─── transition_view ──────────────────────────────────────────


def test_transition_view_keys():
    view = transition_view(
        {"avatar.r": 3.0, "objects.count": 5},
        {"avatar.r": 4.0, "objects.count": 4},
    )
    assert view["avatar.r"] == 3.0
    assert view["avatar.r.after"] == 4.0
    assert view["avatar.r.delta"] == 1.0
    assert view["objects.count.delta"] == -1.0
    assert view["objects.count.changed"] is True


# ─── mechanics lifecycle ──────────────────────────────────────


def test_mechanic_hypothesis_verifies_on_repeated_support():
    ledger = HypothesisLedger()
    # ACTION1 keeps decrementing the count — the world obeys the rule.
    for _ in range(4):
        ledger.abduce_mechanics(
            "1", {"objects.count": 5}, {"objects.count": 4}
        )
        ledger.observe_transition(
            "1", {"objects.count": 4}, {"objects.count": 3}
        )
    hyps = [h for h in ledger.hypotheses.values() if h.kind == "mechanic"]
    assert hyps
    # The delta-rule should be standing and verified.
    delta = [h for h in hyps if h.effect[0].key.endswith(".delta")]
    assert delta and delta[0].status == "verified"


def test_mechanic_hypothesis_falsified_when_effect_absent():
    ledger = HypothesisLedger()
    ledger.abduce_mechanics(
        "1", {"objects.count": 5}, {"objects.count": 4}
    )
    # The "delta == -1" rule then fails three times straight.
    for _ in range(3):
        ledger.observe_transition(
            "1", {"objects.count": 4}, {"objects.count": 4}
        )
    refuted = [
        h for h in ledger.hypotheses.values() if h.status == "refuted"
    ]
    assert refuted


# ─── goal abduction and falsification ─────────────────────────


def test_win_abduces_goal_conditions():
    ledger = HypothesisLedger()
    new = ledger.abduce_terminal(
        True,
        {"objects.count": 0, "avatar.r": 5.0},
        changed_last=("objects.count",),
    )
    assert new
    # The last-changed variable leads the candidates.
    assert new[0].effect[0].key == "objects.count"
    assert ledger.best_goal() is not None


def test_goal_falsified_when_satisfied_without_winning():
    ledger = HypothesisLedger()
    ledger.abduce_terminal(True, {"objects.count": 0})
    # Play continues at count == 0 twice: the condition held and
    # the episode did not end — it was never the goal.
    ledger.observe_continuation({"objects.count": 0})
    ledger.observe_continuation({"objects.count": 0})
    ledger.observe_continuation({"objects.count": 0})
    goal = ledger.best_goal()
    assert goal is None or goal.status == "refuted"


def test_loss_abduces_avoid_and_scores_against_state():
    ledger = HypothesisLedger()
    ledger.abduce_terminal(False, {"hazard.touching": True})
    assert ledger.goal_score({"hazard.touching": True}) < 0.0
    assert ledger.goal_score({"hazard.touching": False}) == 0.0


# ─── TaskCompetence integration ───────────────────────────────


def test_competence_records_and_abduces():
    competence = TaskCompetence()
    ctx = _ctx(competence, {"objects.count": 2, "avatar.r": 1})
    competence.record_transition(
        ctx, "1", {"objects.count": 2}, {"objects.count": 1}
    )
    competence.record_episode(
        ctx,
        steps=(),
        success=True,
        verification_score=1.0,
        verification="external",
        final_state={"objects.count": 0},
        terminal_changes=("objects.count",),
    )
    hyps = competence.hypotheses(ctx)
    kinds = {h.kind for h in hyps}
    assert "mechanic" in kinds and "goal" in kinds
    goals = competence.goal_conditions(ctx)
    assert goals and goals[0].key == "objects.count"


def test_goal_score_and_predicted_gain():
    competence = TaskCompetence()
    ctx = _ctx(competence, {"objects.count": 2})
    # Teach the effect model that ACTION1 decrements the count.
    for _ in range(6):
        competence.record_transition(
            ctx, "1", {"objects.count": 2}, {"objects.count": 1}
        )
    competence.record_episode(
        ctx,
        steps=(),
        success=True,
        verification_score=1.0,
        verification="external",
        final_state={"objects.count": 0},
        terminal_changes=("objects.count",),
    )
    # Goal hyp "objects.count eq 0" standing → satisfaction now and
    # predicted gain for the action that moves toward it.
    assert competence.goal_score(ctx, {"objects.count": 0}) > 0.0
    gain = competence.predicted_goal_gain(
        ctx, "1", {"objects.count": 1}
    )
    assert gain > 0.0


def test_epistemic_bonus_targets_undecided_rules():
    competence = TaskCompetence()
    ctx = _ctx(competence, {"objects.count": 2})
    # One observation abduces a proposed mechanic hypothesis for "1".
    competence.record_transition(
        ctx, "1", {"objects.count": 2}, {"objects.count": 1}
    )
    bonus = competence.epistemic_bonus(ctx, "1", {"objects.count": 2})
    assert bonus > 0.0
    # An action with no hypothesis on trial gets nothing.
    assert competence.epistemic_bonus(ctx, "2", {"objects.count": 2}) == 0.0


def test_hypotheses_transfer_through_schema_matching():
    competence = TaskCompetence()
    ctx = _ctx(competence, {"objects.count": 2})
    competence.record_episode(
        ctx,
        steps=(),
        success=True,
        verification_score=1.0,
        verification="external",
        final_state={"objects.count": 0},
        terminal_changes=("objects.count",),
    )
    # A structurally identical environment recognizes the same
    # schema — and inherits its hypotheses as priors.
    ctx2 = competence.recognize(
        domain="test.env",
        state={"objects.count": 5, "avatar.r": 9},
        actions=("1", "2"),
        roles=("self",),
    )
    assert ctx2.schema is ctx.schema
    assert competence.hypotheses(ctx2)


def test_schema_serialization_carries_hypotheses():
    from frontal_lobe.reasoning.competence import TaskSchema

    competence = TaskCompetence()
    ctx = _ctx(competence, {"objects.count": 2})
    competence.record_transition(
        ctx, "1", {"objects.count": 2}, {"objects.count": 1}
    )
    data = ctx.schema.to_dict()
    restored = TaskSchema.from_dict(data)
    assert restored.hypotheses is not None
    assert restored.hypotheses.hypotheses


def test_ledger_cap_evicts_weakest():
    ledger = HypothesisLedger()
    for i in range(40):
        ledger.add(
            Hypothesis(
                hypothesis_id=f"h{i}",
                kind="goal",
                effect=(GoalCondition(f"k{i}", "eq", i),),
            )
        )
    assert len(ledger.hypotheses) <= 24
