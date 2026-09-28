"""AGI trajectory probe: held-out causal rule generalization.

This probe deliberately tests a boundary of TaskCompetence rather than
constructing a success case. Genesis explores a hidden deterministic
environment, records observed action effects, and is then given a held-out
instance whose concrete target and action names have never been observed.

The probe asks whether the existing competence substrate can abstract:
    observable feature -> successful action role
rather than merely memorizing:
    concrete action name -> observed effect.

A second domain uses disjoint vocabulary. No correct action is exposed to the
agent during selection. A failure to solve the held-out instance is useful
evidence: it identifies the gap between transition-memory and latent rule
induction without disguising that gap with an oracle adapter.

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
class RuleFamily:
    feature_key: str
    true_role: str
    false_role: str


@dataclass(frozen=True)
class Trial:
    domain: str
    target: str
    feature_value: bool
    actions: tuple[str, str]


def make_trial(domain: str, target: str, feature_value: bool) -> Trial:
    prefix = "align" if domain == "lattice" else "index"
    return Trial(
        domain=domain,
        target=target,
        feature_value=feature_value,
        actions=(
            f"{prefix}_{target}",
            f"{prefix}_false_{target}",
        ),
    )


def make_state(trial: Trial) -> dict[str, Any]:
    key = "amber" if trial.domain == "lattice" else "ochre"
    return {key: trial.feature_value}


def apply_hidden_rule(
    family: RuleFamily, trial: Trial, action: str
) -> dict[str, Any]:
    key = family.feature_key
    expected_role = (
        family.true_role if trial.feature_value else family.false_role
    )
    # The environment knows the hidden causal rule. The agent sees only
    # the resulting state and success signal.
    prefix = action.split("_", 1)[0]
    is_false_action = "_false_" in action
    observed_role = "false" if is_false_action else prefix
    success = observed_role == expected_role
    return {
        key: trial.feature_value,
        "solved": success,
    }


def recognize(competence: TaskCompetence, trial: Trial):
    goal = (GoalCondition("solved", "eq", True),)
    return competence.recognize(
        domain=trial.domain,
        state=make_state(trial),
        actions=trial.actions,
        goal_conditions=goal,
        roles=("inspect", "select", "support"),
    )


def explore(
    competence: TaskCompetence,
    family: RuleFamily,
    trials: list[Trial],
) -> list[dict[str, Any]]:
    observations: list[dict[str, Any]] = []
    goal = (GoalCondition("solved", "eq", True),)

    for trial in trials:
        context = recognize(competence, trial)

        # Blind exploration. The environment is the only source of the
        # success signal; no correct_action field exists in this harness.
        for action in trial.actions:
            before = make_state(trial)
            after = apply_hidden_rule(family, trial, action)
            check = competence.record_transition(
                context, action, before, after, success=after["solved"]
            )

            item = {
                "domain": trial.domain,
                "target": trial.target,
                "feature_value": trial.feature_value,
                "action": action,
                "success": after["solved"],
                "prediction_score": check.score,
                "predicted_effects": [
                    p.to_dict() for p in check.predicted
                ],
            }

            if after["solved"]:
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
                    verification_score=competence.verify(goal, after),
                    goal_conditions=goal,
                    verification="external",
                )
                item["skill_created"] = skill is not None
                observations.append(item)
                break

            item["skill_created"] = False
            observations.append(item)

    return observations


def model_select(
    competence: TaskCompetence,
    trial: Trial,
) -> dict[str, Any]:
    """Select using only the currently learned concrete operator models."""
    context = recognize(competence, trial)
    candidates: list[tuple[float, str]] = []

    for action in trial.actions:
        predictions = competence.predict(
            context, action, make_state(trial)
        )
        solved_predictions = [
            p for p in predictions
            if p.key == "solved" and p.expected is True
        ]
        if solved_predictions:
            candidates.append(
                (max(p.confidence for p in solved_predictions), action)
            )

    selected = max(candidates, default=(0.0, None))[1]
    return {
        "selected_action": selected,
        "known_positive_model": selected is not None,
        "context_novel": context.novel,
    }


def execute_selection(
    competence: TaskCompetence,
    family: RuleFamily,
    trial: Trial,
) -> dict[str, Any]:
    selection = model_select(competence, trial)
    action = selection["selected_action"]
    after = (
        apply_hidden_rule(family, trial, action)
        if action is not None
        else {"solved": False}
    )
    return {
        **selection,
        "success": after["solved"],
        "target": trial.target,
        "feature_value": trial.feature_value,
    }


def main() -> None:
    competence = TaskCompetence()
    family = RuleFamily(
        feature_key="amber",
        true_role="align",
        false_role="false",
    )

    training = [
        make_trial("lattice", "ravel", True),
        make_trial("lattice", "sorn", False),
        make_trial("lattice", "tavi", True),
    ]
    observations = explore(competence, family, training)

    held_out = execute_selection(
        competence,
        family,
        make_trial("lattice", "merek", True),
    )

    restored = TaskCompetence.from_dict(competence.to_dict())
    restored_held_out = execute_selection(
        restored,
        family,
        make_trial("lattice", "nema", True),
    )

    archive_family = RuleFamily(
        feature_key="ochre",
        true_role="index",
        false_role="false",
    )
    transfer_trial = make_trial("archive", "ulna", True)
    transfer_context = recognize(competence, transfer_trial)

    report = {
        "probe": "agi-trajectory-rule-generalization-v3",
        "llm_in_loop": False,
        "correct_action_exposed": False,
        "environment_rule_hidden": True,
        "training_observations": observations,
        "held_out": held_out,
        "restored_held_out": restored_held_out,
        "cross_domain": {
            "recognition_novel": transfer_context.novel,
            "similarity": transfer_context.similarity,
            "foreign_skill_count": sum(
                m.skill.signature.domain != "archive"
                for m in transfer_context.skills
            ),
            "transfer_skills": [
                {
                    "skill_id": m.skill.skill_id,
                    "source_domain": m.skill.signature.domain,
                    "similarity": m.similarity,
                    "score": m.score,
                }
                for m in transfer_context.skills
            ],
            "independent_environment_rule": archive_family.true_role,
        },
        "measurements": {
            "training_successes": sum(
                int(x["success"]) for x in observations
            ),
            "learned_skills": competence.skill_count,
            "learned_schemas": competence.schema_count,
            "transition_observations": competence.transition_count,
            "held_out_success": held_out["success"],
            "restored_held_out_success": restored_held_out["success"],
        },
        "interpretation": [
            "Training success establishes that external outcomes can be "
            "recorded as concrete action-effect knowledge.",
            "Held-out success requires abstraction beyond the concrete "
            "operator names observed during training.",
            "A held-out failure is evidence of a rule-induction gap, not "
            "evidence that the environment or the agent is generally "
            "incapable.",
            "Cross-domain retrieval is reported separately because "
            "retrieving a foreign skill is weaker evidence than inferring "
            "a latent causal rule.",
        ],
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
