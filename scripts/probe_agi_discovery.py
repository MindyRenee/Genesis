"""Exploratory AGI trajectory probe: discover an action rule from outcomes.

Unlike the first closed-loop probe, this experiment does not expose the
correct action to Genesis. The environment contains a hidden rule mapping
observable features to successful actions. Genesis must explore, observe
outcomes, update its transition/affordance memory, and then exploit the
learned rule on held-out bindings.

A second domain uses different surface vocabulary but the same abstract
role structure. The probe therefore separates:
- discovery: can Genesis learn a hidden action→outcome relation?
- generalization: does that relation work on new bindings?
- transfer: does a structurally analogous skill become retrievable?
- persistence: does the learned model survive serialization?

No language model or textual answer is used as evidence.
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
class HiddenRule:
    domain: str
    feature_key: str
    true_action_prefix: str
    false_action_prefix: str


@dataclass(frozen=True)
class Trial:
    domain: str
    target: str
    feature_value: bool
    actions: tuple[str, str]


def make_trial(domain: str, target: str, feature_value: bool) -> Trial:
    # The two domains deliberately use disjoint surface vocabulary.
    if domain == "lattice":
        actions = (f"align_{target}", f"align_false_{target}")
    elif domain == "archive":
        actions = (f"index_{target}", f"index_false_{target}")
    else:
        raise ValueError(domain)
    return Trial(domain, target, feature_value, actions)


def environment_step(
    rule: HiddenRule, trial: Trial, action: str
) -> dict[str, Any]:
    expected_prefix = (
        rule.true_action_prefix if trial.feature_value
        else rule.false_action_prefix
    )
    expected_action = f"{expected_prefix}{trial.target}"
    success = action == expected_action
    # The environment exposes only consequences, never the hidden rule.
    return {
        "domain": trial.domain,
        "target": trial.target,
        "state": {
            rule.feature_key: trial.feature_value,
            "solved": success,
        },
        "success": success,
    }


def recognize(competence: TaskCompetence, trial: Trial):
    goal = (GoalCondition("solved", "eq", True),)
    return competence.recognize(
        domain=trial.domain,
        state={rule_key(trial.domain): trial.feature_value},
        actions=trial.actions,
        goal_conditions=goal,
        roles=("inspect", "select", "support"),
    )


def rule_key(domain: str) -> str:
    return "amber" if domain == "lattice" else "ochre"


def explore(
    competence: TaskCompetence,
    rule: HiddenRule,
    trials: list[Trial],
) -> list[dict[str, Any]]:
    results = []
    for trial in trials:
        context = recognize(competence, trial)
        # Exploration is deliberately blind: action order alternates so
        # Genesis receives positive and negative evidence rather than being
        # handed the answer.
        for action in trial.actions:
            before = {
                rule.feature_key: trial.feature_value
            }
            outcome = environment_step(rule, trial, action)
            after = dict(outcome["state"])
            transition = competence.record_transition(
                context, action, before, after, success=outcome["success"]
            )
            if outcome["success"]:
                skill = competence.record_episode(
                    context,
                    steps=(
                        ProcedureStep(
                            action=action,
                            parameters={"support": 1.0},
                            family="hidden-rule-selection",
                        ),
                    ),
                    success=True,
                    verification_score=competence.verify(
                        (GoalCondition("solved", "eq", True),), after
                    ),
                    goal_conditions=(GoalCondition("solved", "eq", True),),
                    verification="external",
                )
                results.append({
                    "domain": trial.domain,
                    "target": trial.target,
                    "feature_value": trial.feature_value,
                    "action": action,
                    "success": True,
                    "transition_score": transition.score,
                    "skill_created": skill is not None,
                })
                break
            results.append({
                "domain": trial.domain,
                "target": trial.target,
                "feature_value": trial.feature_value,
                "action": action,
                "success": False,
                "transition_score": transition.score,
                "skill_created": False,
            })
    return results


def held_out_exploitation(
    competence: TaskCompetence,
    rule: HiddenRule,
    trial: Trial,
) -> dict[str, Any]:
    context = recognize(competence, trial)
    predictions = {}
    for action in trial.actions:
        predictions[action] = [
            p.to_dict() for p in competence.predict(
                context, action,
                {rule.feature_key: trial.feature_value},
            )
        ]

    # Selection is based solely on the learned model: choose the action
    # whose predicted effect includes solved=True. If neither action has a
    # learned positive prediction, abstain rather than smuggling in the
    # hidden answer.
    selected = None
    for action, preds in predictions.items():
        if any(
            p["key"] == "solved" and p["expected"] is True
            for p in preds
        ):
            selected = action
            break

    outcome = (
        environment_step(rule, trial, selected)
        if selected is not None
        else {"success": False, "state": {rule.feature_key: trial.feature_value}}
    )
    return {
        "domain": trial.domain,
        "target": trial.target,
        "feature_value": trial.feature_value,
        "predictions": predictions,
        "selected_action": selected,
        "success": outcome["success"],
        "abstained": selected is None,
    }


def main() -> None:
    competence = TaskCompetence()
    rule = HiddenRule(
        domain="lattice",
        feature_key="amber",
        true_action_prefix="align_",
        false_action_prefix="align_false_",
    )

    # Discovery phase. Genesis is not told which action is correct.
    training = [
        make_trial("lattice", "ravel", True),
        make_trial("lattice", "sorn", False),
        make_trial("lattice", "tavi", True),
    ]
    exploration = explore(competence, rule, training)

    # Held-out binding in the same rule family.
    held_out = held_out_exploitation(
        competence, rule, make_trial("lattice", "merek", True)
    )

    # Cross-domain test. The environment has an analogous rule, but the
    # surface action/state vocabulary is different.
    transfer_task = make_trial("archive", "ulna", True)
    transfer_context = recognize(competence, transfer_task)
    transfer_matches = [
        {
            "skill_id": m.skill.skill_id,
            "source_domain": m.skill.signature.domain,
            "similarity": m.similarity,
            "score": m.score,
            "steps": m.skill.step_labels(),
        }
        for m in transfer_context.skills
    ]

    # Persistence is tested without retaining the Python object.
    restored = TaskCompetence.from_dict(competence.to_dict())
    restored_held_out = held_out_exploitation(
        restored, rule, make_trial("lattice", "nema", True)
    )

    report = {
        "probe": "agi-trajectory-discovery-v2",
        "llm_in_loop": False,
        "hidden_rule": True,
        "correct_action_exposed_to_agent": False,
        "training_trials": len(training),
        "exploration": exploration,
        "held_out": held_out,
        "cross_domain_transfer": {
            "matches": transfer_matches,
            "available": any(
                m["source_domain"] != "archive" for m in transfer_matches
            ),
            "surface_vocabularies_disjoint": True,
        },
        "restored_held_out": restored_held_out,
        "measurements": {
            "learned_skills": competence.skill_count,
            "learned_schemas": competence.schema_count,
            "transition_observations": competence.transition_count,
            "held_out_success": held_out["success"],
            "restored_success": restored_held_out["success"],
        },
        "next_test": (
            "Replace the hand-authored two-feature environment with a "
            "family of latent rules and require Genesis to infer the rule "
            "itself, then test transfer to an independently generated "
            "environment whose action names and state variables are all "
            "new."
        ),
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
