"""AGI trajectory probe: closed-loop learning curve without an action oracle.

This benchmark isolates sample efficiency before asking for abstract transfer.
The environment hides which of two operators succeeds. Operator names are
reused across trials, while bindings and feature values vary. Genesis must
explore unknown operators, record observed outcomes, and then exploit a
positive learned effect model.

The probe reports the number of environment interactions required to solve
each trial and the prediction accuracy before acting. It deliberately does
not treat a successful trial as evidence of abstraction: that is tested by
the separate rule-generalization probe.

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
class Trial:
    target: str
    actions: tuple[str, str]
    correct_index: int


def make_trial(target: str, correct_index: int) -> Trial:
    return Trial(
        target=target,
        actions=("probe_left", "probe_right"),
        correct_index=correct_index,
    )


def choose_action(
    competence: TaskCompetence,
    context,
    trial: Trial,
    *,
    exploration_cursor: int,
) -> str:
    """Exploit a learned positive effect, otherwise explore deterministically."""
    candidates: list[tuple[float, str]] = []
    state = {"target.id": trial.target}

    for action in trial.actions:
        predictions = competence.predict(context, action, state)
        for prediction in predictions:
            if prediction.key == "solved" and prediction.expected is True:
                candidates.append((prediction.confidence, action))

    if candidates:
        return max(candidates, key=lambda item: (item[0], item[1]))[1]
    return trial.actions[exploration_cursor % len(trial.actions)]


def run_trial(
    competence: TaskCompetence,
    trial: Trial,
    *,
    exploration_cursor: int,
) -> dict[str, Any]:
    goal = (GoalCondition("solved", "eq", True),)
    context = competence.recognize(
        domain="hidden-choice",
        state={"target.id": trial.target},
        actions=trial.actions,
        goal_conditions=goal,
        roles=("inspect", "select", "support"),
    )

    attempts = 0
    records: list[dict[str, Any]] = []
    while attempts < len(trial.actions):
        action = choose_action(
            competence, context, trial,
            exploration_cursor=exploration_cursor + attempts,
        )
        before = {"target.id": trial.target}
        success = action == trial.actions[trial.correct_index]
        after = {**before, "solved": success}
        check = competence.record_transition(
            context, action, before, after, success=success
        )
        attempts += 1
        records.append({
            "action": action,
            "success": success,
            "prediction_score": check.score,
            "predicted_effects": [p.to_dict() for p in check.predicted],
        })

        if success:
            competence.record_episode(
                context,
                steps=(
                    ProcedureStep(
                        action=action,
                        parameters={"support": 1.0},
                        family="hidden-choice",
                    ),
                ),
                success=True,
                verification_score=competence.verify(goal, after),
                goal_conditions=goal,
                verification="external",
            )
            break

    return {
        "target": trial.target,
        "attempts": attempts,
        "success": records[-1]["success"],
        "records": records,
    }


def main() -> None:
    competence = TaskCompetence()

    # The concrete operator identity is stable, but target bindings change.
    # The first trial requires exploration; later trials should become cheaper
    # if the learned operator model is actually being used.
    trials = [
        make_trial("a", 0),
        make_trial("b", 0),
        make_trial("c", 0),
        make_trial("d", 0),
        make_trial("e", 0),
    ]

    episodes = [
        run_trial(
            competence,
            trial,
            exploration_cursor=index,
        )
        for index, trial in enumerate(trials)
    ]

    report = {
        "probe": "agi-trajectory-learning-curve-v1",
        "llm_in_loop": False,
        "correct_action_exposed": False,
        "environment_rule_hidden": True,
        "episodes": episodes,
        "measurements": {
            "total_environment_interactions": sum(
                e["attempts"] for e in episodes
            ),
            "attempts_per_episode": [e["attempts"] for e in episodes],
            "all_successful": all(e["success"] for e in episodes),
            "transition_observations": competence.transition_count,
            "learned_skills": competence.skill_count,
        },
        "interpretation": [
            "A decreasing interaction cost indicates concrete affordance "
            "learning and exploitation.",
            "This does not establish abstraction: operator names are reused.",
            "To establish abstraction, the rule-generalization probe changes "
            "the concrete operators and requires a latent rule to transfer.",
        ],
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
