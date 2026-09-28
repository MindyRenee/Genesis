"""AGI trajectory probe: meta-learning across independently named task families.

The question here is not whether Genesis can solve one task. It is whether
experience with one family changes the *cost of learning another family*.

Two deterministic environments implement the same abstract binary decision
structure but use disjoint feature and action vocabularies. A fresh agent and
a pretrained agent receive the same novel family. Neither is told the hidden
rule or the correct action. Each must discover the rule through environmental
feedback.

The benchmark is intentionally diagnostic: if pretrained Genesis does not
learn the second family faster, that is evidence that current competence
memory has not yet become a reusable learning prior across this degree of
surface variation.

No LLM is used.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from genesis_cognitive.reasoning.competence import (
    GoalCondition,
    ProcedureStep,
    TaskCompetence,
)


@dataclass(frozen=True)
class Family:
    domain: str
    feature_key: str
    actions: tuple[str, str]
    correct_for_true: int
    correct_for_false: int


@dataclass(frozen=True)
class Trial:
    family: Family
    target: str
    feature_value: bool


def make_trial(family: Family, target: str, feature_value: bool) -> Trial:
    return Trial(family=family, target=target, feature_value=feature_value)


def step(family: Family, trial: Trial, action: str) -> dict[str, Any]:
    index = family.correct_for_true if trial.feature_value else family.correct_for_false
    return {
        family.feature_key: trial.feature_value,
        "solved": action == family.actions[index],
    }


def context_for(competence: TaskCompetence, trial: Trial):
    return competence.recognize(
        domain=trial.family.domain,
        state={trial.family.feature_key: trial.feature_value},
        actions=trial.family.actions,
        goal_conditions=(GoalCondition("solved", "eq", True),),
        roles=("inspect", "select", "support"),
    )


def discover(
    competence: TaskCompetence,
    trials: list[Trial],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []

    for trial in trials:
        context = context_for(competence, trial)

        # Explore each currently unknown operator exactly once. The agent
        # receives no hidden-rule metadata and no correct-action hint.
        for action in trial.family.actions:
            before = {trial.family.feature_key: trial.feature_value}
            after = step(trial.family, trial, action)
            check = competence.record_transition(
                context,
                action,
                before,
                after,
                success=after["solved"],
            )
            records.append({
                "domain": trial.family.domain,
                "target": trial.target,
                "feature_value": trial.feature_value,
                "action": action,
                "success": after["solved"],
                "prediction_score": check.score,
            })

            if after["solved"]:
                competence.record_episode(
                    context,
                    steps=(
                        ProcedureStep(
                            action=action,
                            parameters={"support": 1.0},
                            family="binary-rule",
                        ),
                    ),
                    success=True,
                    verification_score=competence.verify(
                        (GoalCondition("solved", "eq", True),), after
                    ),
                    goal_conditions=(GoalCondition("solved", "eq", True),),
                    verification="external",
                )
                break

    return records


def run_trial(
    competence: TaskCompetence,
    trial: Trial,
    *,
    exploration_cursor: int,
) -> dict[str, Any]:
    """Run a target trial closed-loop; no correct-action oracle reaches Genesis."""
    goal = (GoalCondition("solved", "eq", True),)
    context = context_for(competence, trial)
    state = {trial.family.feature_key: trial.feature_value}
    records: list[dict[str, Any]] = []

    for offset in range(len(trial.family.actions)):
        predicted: list[tuple[float, str]] = []
        for action in trial.family.actions:
            for effect in competence.predict(context, action, state):
                if effect.key == "solved" and effect.expected is True:
                    predicted.append((effect.confidence, action))

        if predicted:
            action = max(predicted, key=lambda item: (item[0], item[1]))[1]
            mode = "model"
        else:
            action = trial.family.actions[
                (exploration_cursor + offset) % len(trial.family.actions)
            ]
            mode = "explore"

        after = step(trial.family, trial, action)
        check = competence.record_transition(
            context, action, state, after, success=after["solved"]
        )
        records.append({
            "action": action,
            "mode": mode,
            "success": after["solved"],
            "prediction_score": check.score,
        })

        if after["solved"]:
            competence.record_episode(
                context,
                steps=(
                    ProcedureStep(
                        action=action,
                        parameters={"support": 1.0},
                        family="binary-rule",
                    ),
                ),
                success=True,
                verification_score=competence.verify(goal, after),
                goal_conditions=goal,
                verification="external",
            )
            break

    return {
        "domain": trial.family.domain,
        "target": trial.target,
        "feature_value": trial.feature_value,
        "attempts": len(records),
        "success": records[-1]["success"],
        "records": records,
        "cross_domain_retrievals": [
            {
                "skill_id": match.skill.skill_id,
                "source_domain": match.skill.signature.domain,
                "similarity": match.similarity,
                "score": match.score,
            }
            for match in context.skills
            if match.skill.signature.domain != trial.family.domain
        ],
    }


def main() -> None:
    source = Family(
        domain="lattice",
        feature_key="amber",
        actions=("align", "reject"),
        correct_for_true=0,
        correct_for_false=1,
    )
    target = Family(
        domain="archive",
        feature_key="ochre",
        actions=("index", "discard"),
        correct_for_true=0,
        correct_for_false=1,
    )

    source_trials = [
        make_trial(source, "ravel", True),
        make_trial(source, "sorn", False),
    ]
    target_trials = [
        make_trial(target, "ulna", True),
        make_trial(target, "ves", False),
        make_trial(target, "keth", True),
        make_trial(target, "mora", False),
    ]

    fresh = TaskCompetence()
    baseline_records = [
        run_trial(fresh, trial, exploration_cursor=index)
        for index, trial in enumerate(target_trials)
    ]

    pretrained = TaskCompetence()
    source_records = discover(pretrained, source_trials)
    pretrained_records = [
        run_trial(pretrained, trial, exploration_cursor=index)
        for index, trial in enumerate(target_trials)
    ]

    report = {
        "probe": "agi-trajectory-meta-learning-v2",
        "llm_in_loop": False,
        "correct_action_exposed": False,
        "environment_rule_hidden": True,
        "surface_vocabularies_disjoint": True,
        "source_learning": source_records,
        "fresh_target": baseline_records,
        "pretrained_target": pretrained_records,
        "measurements": {
            "fresh_interactions": sum(e["attempts"] for e in baseline_records),
            "pretrained_source_interactions": len(source_records),
            "pretrained_target_interactions": sum(
                e["attempts"] for e in pretrained_records
            ),
            "fresh_successes": sum(e["success"] for e in baseline_records),
            "pretrained_successes": sum(
                e["success"] for e in pretrained_records
            ),
            "fresh_model_selections": sum(
                r["mode"] == "model"
                for e in baseline_records for r in e["records"]
            ),
            "pretrained_model_selections": sum(
                r["mode"] == "model"
                for e in pretrained_records for r in e["records"]
            ),
            "pretrained_cross_domain_retrievals": sum(
                bool(e["cross_domain_retrievals"]) for e in pretrained_records
            ),
            "pretrained_cross_domain_retrievals_used_for_action": 0,
        },
        "interpretation": [
            "The fresh and pretrained arms solve the same target sequence without "
            "being given the hidden rule or correct action.",
            "A reduction in target interactions for the pretrained arm would be "
            "evidence that source experience changes target learning cost.",
            "Cross-domain retrieval is measured separately because retrieval is "
            "not equivalent to rebinding a foreign procedure to target operators.",
            "The current TaskCompetence.predict() API predicts effects for named "
            "operators stored in the target schema; it does not consume a foreign "
            "skill as an action-selection rule.",
            "This probe therefore distinguishes stored skill retrieval from a "
            "reusable meta-learning prior and does not claim general intelligence.",
        ],
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
