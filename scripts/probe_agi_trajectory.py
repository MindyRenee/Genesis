"""Closed-loop AGI trajectory probe for Genesis's non-LLM competence substrate.

This probe treats Genesis as an agent interacting with a deterministic
environment, not as a language model. It measures whether experience changes
later action selection, whether the learned transition model predicts novel
instances, whether a verified procedure transfers across disjoint domains,
and whether the learned competence survives serialization.

The environment has two domains with deliberately disjoint surface
vocabularies. Both expose the same abstract task schema: inspect a target,
select the supported option, and reach a terminal condition. The probe does
not call an LLM and does not accept textual answers as evidence.

This is a mechanism-level AGI probe, not an AGI claim. Passing it establishes
that the tested competence substrate exhibits learning/transfer behavior;
it does not establish general intelligence.
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
class ToyTask:
    domain: str
    state: dict[str, Any]
    actions: tuple[str, ...]
    target: str
    correct_action: str


def make_task(
    domain: str,
    *,
    target: str,
    correct_action: str,
    distractor_action: str,
) -> ToyTask:
    # Surface vocabulary is intentionally disjoint between domains.
    if domain == "lattice":
        state = {"node": target, "amber": True, "violet": False}
    elif domain == "archive":
        state = {"folio": target, "ochre": True, "indigo": False}
    else:
        raise ValueError(domain)
    return ToyTask(
        domain=domain,
        state=state,
        actions=(correct_action, distractor_action),
        target=target,
        correct_action=correct_action,
    )


def run_episode(
    competence: TaskCompetence,
    task: ToyTask,
    *,
    learn: bool,
    verification: str = "external",
) -> dict[str, Any]:
    """Run a closed-loop episode without exposing the environment oracle."""
    goal = (GoalCondition("solved", "eq", True),)
    context = competence.recognize(
        domain=task.domain,
        state=task.state,
        actions=task.actions,
        goal_conditions=goal,
        roles=("inspect", "select", "support"),
    )

    records: list[dict[str, Any]] = []
    tried: set[str] = set()

    for _ in task.actions:
        predicted = []
        for action in task.actions:
            if action in tried:
                continue
            for effect in competence.predict(context, action, task.state):
                if effect.key == "solved" and effect.expected is True:
                    predicted.append((effect.confidence, action))

        if predicted:
            selected = max(predicted, key=lambda item: (item[0], item[1]))[1]
            mode = "model"
        else:
            selected = next(action for action in task.actions if action not in tried)
            mode = "explore"

        tried.add(selected)
        before = dict(task.state)
        after = dict(before)
        after["solved"] = selected == task.correct_action

        transition = competence.record_transition(
            context,
            selected,
            before,
            after,
            success=after["solved"],
        )
        record = {
            "action": selected,
            "mode": mode,
            "success": after["solved"],
            "transition_score": transition.score,
            "predictions_before_action": [
                {
                    "key": p.key,
                    "kind": p.kind,
                    "expected": p.expected,
                    "confidence": p.confidence,
                }
                for p in transition.predicted
            ],
        }
        records.append(record)

        if after["solved"]:
            if learn:
                skill = competence.record_episode(
                    context,
                    steps=(
                        ProcedureStep(
                            action=selected,
                            parameters={"support": 1.0},
                            family="constraint-selection",
                        ),
                    ),
                    success=True,
                    verification_score=competence.verify(goal, after),
                    goal_conditions=goal,
                    verification=verification,
                )
                record["skill_created"] = skill is not None
            break

    return {
        "novel": context.novel,
        "schema_id": context.schema.schema_id,
        "attempts": len(records),
        "selected_action": records[-1]["action"],
        "success": records[-1]["success"],
        "records": records,
        "skill_count": competence.skill_count,
    }


def transfer_episode(
    competence: TaskCompetence,
    task: ToyTask,
) -> dict[str, Any]:
    """Measure cross-domain retrieval without an oracle-backed action choice."""
    goal = (GoalCondition("solved", "eq", True),)
    context = competence.recognize(
        domain=task.domain,
        state=task.state,
        actions=task.actions,
        goal_conditions=goal,
        roles=("inspect", "select", "support"),
    )
    matches = [
        {
            "skill_id": match.skill.skill_id,
            "similarity": match.similarity,
            "score": match.score,
            "source_domain": match.skill.signature.domain,
            "steps": match.skill.step_labels(),
            "verification": match.skill.verification,
        }
        for match in context.skills
    ]
    foreign = [
        match for match in context.skills
        if match.skill.signature.domain != task.domain
    ]

    # Do not substitute the environment's correct_action here. That would
    # turn retrieval into an oracle-assisted transfer result. Action rebinding
    # is a separate capability and is reported as unavailable to this probe.
    return {
        "novel": context.novel,
        "recognized_similarity": context.similarity,
        "foreign_skill_matches": matches,
        "transfer_available": bool(foreign),
        "action_rebinding_tested": False,
        "success": None,
        "goal": str(goal[0]),
    }


def main() -> None:
    competence = TaskCompetence()

    # Phase 1: acquire a verified skill in domain A.
    source = make_task(
        "lattice",
        target="ravel",
        correct_action="align_ravel",
        distractor_action="align_kelm",
    )
    acquisition = run_episode(competence, source, learn=True)

    # Phase 2: repeat the same structural task with different bindings.
    repeated = make_task(
        "lattice",
        target="sorn",
        correct_action="align_sorn",
        distractor_action="align_vex",
    )
    repetition = run_episode(competence, repeated, learn=True)

    # Phase 3: cross-domain task with disjoint vocabulary.
    target = make_task(
        "archive",
        target="ulna",
        correct_action="index_ulna",
        distractor_action="index_brek",
    )
    transfer = transfer_episode(competence, target)

    # Phase 4: serialize and restore the competence substrate. This tests
    # learned competence rather than merely the in-memory object.
    restored = TaskCompetence.from_dict(competence.to_dict())
    restored_transfer = transfer_episode(restored, target)

    report = {
        "probe": "agi-trajectory-closed-loop-v1",
        "agent_interface": "structured state/action/environment outcome",
        "llm_in_loop": False,
        "domains": ["lattice", "archive"],
        "surface_vocabularies_disjoint": True,
        "source_skill_verification": "external",
        "acquisition": acquisition,
        "repetition": repetition,
        "transfer": transfer,
        "restored_transfer": restored_transfer,
        "measurements": {
            "learned_skills": competence.skill_count,
            "learned_schemas": competence.schema_count,
            "transitions": competence.transition_count,
            "transfer_survived_serialization": (
                restored_transfer["transfer_available"]
            ),
            "transfer_retrieval_available": transfer["transfer_available"],
            "restored_transfer_retrieval_available": (
                restored_transfer["transfer_available"]
            ),
        },
        "interpretation": [
            "A positive acquisition result means the competence substrate "
            "learned an externally verified procedure through environment "
            "feedback.",
            "This does not establish AGI: the environment is deterministic "
            "and deliberately small.",
            "A stronger subsequent probe should vary task rules, hide the "
            "correct action, require exploration, introduce failures, and "
            "measure whether Genesis discovers the transferable abstraction "
            "rather than receiving it through the adapter.",
        ],
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
