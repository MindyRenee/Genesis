"""Task competence — schema recognition, affordances, and skill reuse."""

from frontal_lobe.reasoning import (
    GoalCondition,
    ProcedureStep,
    TaskCompetence,
    TaskContext,
)
from self import Reflection, SelfModel
from temporal_lobe.concepts import ConceptNetwork

from infrastructure.narrative import Narrator
from infrastructure.persistence import (
    load_state,
    restore_task_competence,
    save_state,
)


def _context(
    competence: TaskCompetence,
    *,
    position: int = 1,
    distance: int = 3,
) -> TaskContext:
    return competence.recognize(
        domain="navigation",
        state={
            "agent.position": position,
            "target.distance": distance,
            "agent.present": True,
            "target.present": True,
        },
        actions=("move", "wait"),
        goal_conditions=[GoalCondition("target.distance", "eq", 0)],
        entities=("agent", "target"),
    )


class TestTaskCompetence:
    def test_recognizes_novel_then_similar_task(self):
        competence = TaskCompetence()
        first = _context(competence, position=1)
        second = _context(competence, position=4)

        assert first.novel
        assert not second.novel
        assert second.schema is first.schema
        assert second.similarity >= competence.match_threshold

    def test_learns_action_effect_from_transition(self):
        competence = TaskCompetence()
        context = _context(competence)

        first = competence.record_transition(
            context,
            "move",
            {"agent.position": 1, "target.distance": 3},
            {"agent.position": 2, "target.distance": 2},
            success=True,
        )
        assert first.score < 1.0  # the first change was unpredicted

        second = competence.record_transition(
            context,
            "move",
            {"agent.position": 2, "target.distance": 2},
            {"agent.position": 3, "target.distance": 1},
            success=True,
        )
        assert second.score == 1.0
        predicted = competence.predict(
            context,
            "move",
            {"agent.position": 3, "target.distance": 1},
        )
        by_key = {p.key: p for p in predicted}
        assert by_key["agent.position"].expected == 4
        assert by_key["target.distance"].expected == 0
        assert by_key["agent.position"].confidence > 0.0

    def test_skill_provenance_upgrades_only_on_stronger_evidence(self):
        competence = TaskCompetence()
        context = _context(competence)
        steps = [ProcedureStep("move", {"to": "target"})]

        skill = competence.record_episode(
            context,
            steps=steps,
            success=True,
            verification_score=1.0,
            verification="solver",
        )
        assert skill is not None
        assert skill.verification == "solver"

        # A weaker re-verification must not downgrade the record.
        same = competence.record_episode(
            context,
            steps=steps,
            success=True,
            verification_score=1.0,
            verification="reported",
        )
        assert same is skill
        assert skill.verification == "solver"

        # Stronger evidence upgrades it.
        competence.record_episode(
            context,
            steps=steps,
            success=True,
            verification_score=1.0,
            verification="external",
        )
        assert skill.verification == "external"

    def test_verification_provenance_survives_persistence(self):
        competence = TaskCompetence()
        context = _context(competence)
        competence.record_episode(
            context,
            steps=[ProcedureStep("move")],
            success=True,
            verification_score=1.0,
            verification="external",
        )
        restored = TaskCompetence.from_dict(competence.to_dict())
        skill = next(iter(restored.skills.values()))
        assert skill.verification == "external"

        # Skills saved before provenance existed honestly load as
        # "reported", not as world-verified.
        data = competence.to_dict()
        for s in data["skills"]:
            del s["verification"]
        legacy = TaskCompetence.from_dict(data)
        assert next(iter(legacy.skills.values())).verification == "reported"

    def test_cross_domain_transfer_on_shared_structure(self):
        """A skill learned in one domain is a prior in another.

        When the structural vocabulary aligns — same state features,
        actions, goal shape, roles — the domain label is a discount
        (×0.72), not a wall. This is analogical transfer: "a maze is
        a navigation task wearing different clothes".
        """
        competence = TaskCompetence()
        ctx_a = competence.recognize(
            domain="arcade.navigation",
            state={
                "self.r": 5,
                "self.c": 5,
                "target.distance": 8,
                "objects.count": 3,
            },
            actions=("up", "down", "left", "right"),
            goal_conditions=[GoalCondition("target.distance", "eq", 0)],
            entities=("self", "object", "mover"),
            roles=("navigate", "avoid", "self"),
        )
        skill = competence.record_episode(
            ctx_a,
            steps=[ProcedureStep("approach", {"target": "nearest"})],
            success=True,
            verification_score=1.0,
            verification="external",
        )
        assert skill is not None

        # Same task family, different engine: "tiles.maze" instead of
        # "arcade.navigation" — analogous, not identical.
        ctx_b = competence.recognize(
            domain="tiles.maze",
            state={
                "self.r": 1,
                "self.c": 1,
                "target.distance": 5,
                "objects.count": 4,
            },
            actions=("up", "down", "left", "right"),
            goal_conditions=[GoalCondition("target.distance", "eq", 0)],
            entities=("self", "object", "threat"),
            roles=("navigate", "avoid", "self"),
        )
        assert not ctx_b.novel
        assert ctx_b.schema is ctx_a.schema
        assert ctx_b.similarity < 1.0  # domain discount applied
        assert any(m.skill is skill for m in ctx_b.skills)

    def test_disjoint_vocabulary_does_not_transfer(self):
        """Roles alone can't bridge tasks with no shared vocabulary —
        analogy needs more than one overlapping tag."""
        competence = TaskCompetence()
        ctx_a = competence.recognize(
            domain="arcade.navigation",
            state={"self.r": 5, "target.distance": 8},
            actions=("up", "down"),
            goal_conditions=[GoalCondition("target.distance", "eq", 0)],
            roles=("navigate", "self"),
        )
        competence.record_episode(
            ctx_a,
            steps=[ProcedureStep("approach")],
            success=True,
            verification_score=1.0,
            verification="external",
        )

        far = competence.recognize(
            domain="concept.planning",
            state={"step.status": "ok", "plan.progress": 0.5},
            actions=("understand", "verify"),
            goal_conditions=[GoalCondition("plan.done", "truthy")],
            roles=("navigate",),  # one shared tag is not enough
        )
        assert far.novel
        assert not far.skills

    def test_world_verified_skill_outranks_solver_verified(self):
        competence = TaskCompetence()
        context = _context(competence)
        epistemic = competence.record_episode(
            context,
            steps=[ProcedureStep("move", {"how": "thought"})],
            success=True,
            verification_score=1.0,
            verification="solver",
        )
        worldly = competence.record_episode(
            context,
            steps=[ProcedureStep("move", {"how": "acted"})],
            success=True,
            verification_score=1.0,
            verification="external",
        )
        assert epistemic is not None and worldly is not None

        matches = competence.skill_matches(context.signature)
        assert matches[0].skill is worldly
        assert matches[0].score > matches[1].score

    def test_mismatched_observation_is_reported_not_rewritten(self):
        competence = TaskCompetence()
        context = _context(competence)
        competence.record_transition(
            context,
            "move",
            {"agent.position": 1},
            {"agent.position": 2},
        )
        competence.record_transition(
            context,
            "move",
            {"agent.position": 2},
            {"agent.position": 3},
        )

        check = competence.record_transition(
            context,
            "move",
            {"agent.position": 3},
            {"agent.position": 5},
        )
        assert check.score < 1.0
        assert any("agent.position" in m for m in check.mismatches)

    def test_goal_conditions_verify_external_state(self):
        competence = TaskCompetence()
        goals = [
            GoalCondition("target.distance", "eq", 0),
            GoalCondition("agent.alive", "truthy"),
        ]
        assert competence.verify(goals, {"target.distance": 0, "agent.alive": True}) == 1.0
        assert competence.verify(goals, {"target.distance": 2, "agent.alive": True}) < 1.0

    def test_failed_episode_does_not_become_skill(self):
        competence = TaskCompetence()
        context = _context(competence)
        result = competence.record_episode(
            context,
            steps=[ProcedureStep("move", {"delta": 1})],
            success=False,
            verification_score=0.4,
        )
        assert result is None
        assert competence.skill_count == 0
        assert context.schema.failures == 1

    def test_verified_episode_retrieves_on_similar_task(self):
        competence = TaskCompetence()
        context = _context(competence, position=1, distance=3)
        step = ProcedureStep(
            "move",
            {"delta": 1},
            family="move",
            description="move(delta=1)",
        )
        skill = competence.record_episode(
            context,
            steps=[step],
            success=True,
            verification_score=1.0,
            goal_conditions=[GoalCondition("target.distance", "eq", 0)],
        )
        assert skill is not None

        similar = _context(competence, position=5, distance=2)
        assert similar.skills
        assert similar.skills[0].skill is skill
        # Steps keep their caller-supplied content but gain the
        # action's ordinal in the schema's affordance list — the
        # positional metadata positional rebinding uses when a
        # foreign-domain skill is recalled under different action
        # names.
        stored = similar.skills[0].skill.steps[0]
        assert stored.action == step.action
        assert stored.parameters == step.parameters
        assert stored.position == 0

    def test_competence_state_round_trips(self):
        competence = TaskCompetence()
        context = _context(competence)
        competence.record_transition(
            context,
            "move",
            {"agent.position": 1, "target.distance": 3},
            {"agent.position": 2, "target.distance": 2},
            success=True,
        )
        competence.record_episode(
            context,
            steps=[ProcedureStep("move", {"delta": 1})],
            success=True,
            verification_score=1.0,
        )

        restored = TaskCompetence.from_dict(competence.to_dict())
        assert restored.schema_count == competence.schema_count
        assert restored.skill_count == 1
        op = next(iter(restored.schemas.values())).operators["move"]
        assert op.effects["agent.position"].changed == 1

    def test_persistence_round_trip(self, tmp_path):
        competence = TaskCompetence()
        context = _context(competence)
        competence.record_episode(
            context,
            steps=[ProcedureStep("move", {"delta": 1})],
            success=True,
            verification_score=1.0,
        )
        network = ConceptNetwork()
        self_model = SelfModel()
        save_state(
            str(tmp_path),
            network,
            Reflection(network),
            Narrator(self_model, network),
            self_model,
            task_competence=competence,
        )
        data = load_state(str(tmp_path))
        assert data is not None
        restored = TaskCompetence()
        restore_task_competence(restored, data["task_competence"])
        assert restored.skill_count == 1

    def test_verified_skill_is_grounded_in_concept_network(self):
        from temporal_lobe.concepts import ConceptNetwork, RelationType

        net = ConceptNetwork()
        competence = TaskCompetence(network=net)
        context = competence.recognize(
            domain="navigation",
            state={
                "agent.position": 1,
                "target.distance": 3,
                "agent.present": True,
            },
            actions=("move", "wait"),
            goal_conditions=[GoalCondition("target.distance", "eq", 0)],
        )
        skill = competence.record_episode(
            context,
            steps=[ProcedureStep("move"), ProcedureStep("align")],
            success=True,
            verification_score=1.0,
            goal_conditions=[GoalCondition("target.distance", "eq", 0)],
            verification="external",
        )
        assert skill is not None

        concept = net.get_concept(f"skill:{skill.skill_id}")
        assert concept is not None
        assert concept.properties["kind"] == "skill"
        assert concept.properties["verification"] == "external"

        edges = net.get_edges(f"skill:{skill.skill_id}")
        rels = {e.relation for e in edges}
        assert RelationType.PART_OF in rels
        assert RelationType.GOAL_DIRECTED in rels
        assert net.get_concept("domain:navigation") is not None
        assert net.get_concept("goal:target.distance|eq|0") is not None


def _planning_network() -> ConceptNetwork:
    """fire depends on oxygen and fuel; oxygen causes circulation."""
    from temporal_lobe.concepts import RelationType

    net = ConceptNetwork()
    for c in ("fire", "heat", "oxygen", "fuel", "circulation"):
        net.add_concept(c)
    net.add_edge("fire", "heat", RelationType.CAUSES, weight=0.8)
    net.add_edge("oxygen", "fire", RelationType.ENABLES, weight=0.7)
    net.add_edge("fire", "oxygen", RelationType.DEPENDS_ON, weight=0.7)
    net.add_edge("fuel", "fire", RelationType.ENABLES, weight=0.6)
    net.add_edge("fire", "fuel", RelationType.DEPENDS_ON, weight=0.6)
    net.add_edge("oxygen", "circulation", RelationType.CAUSES, weight=0.5)
    return net


def _run_plan(planner, plan, *, succeed: bool = True) -> None:
    while True:
        step = planner.advance(plan)
        if step is None:
            break
        planner.mark_step(plan, step, success=succeed, confidence=0.8)


class TestPlanningCompetenceAdapter:
    def test_step_outcomes_feed_transition_model(self):
        from frontal_lobe.reasoning import Planner

        competence = TaskCompetence()
        planner = Planner(_planning_network(), task_competence=competence)
        plan = planner.create_plan("fire", "understand")
        step = planner.advance(plan)
        assert step is not None
        planner.mark_step(plan, step, success=True, confidence=0.8)

        assert competence.transition_count == 1
        op = plan.task_context.schema.operators[step.step_type]
        assert op.successes == 1
        assert op.effects["step.status"].changed == 1

    def test_completed_plan_consolidates_procedure(self):
        from frontal_lobe.reasoning import Planner, PlanStatus

        competence = TaskCompetence()
        planner = Planner(_planning_network(), task_competence=competence)
        plan = planner.create_plan("fire", "understand")
        _run_plan(planner, plan)

        assert plan.status == PlanStatus.COMPLETED
        assert competence.skill_count == 1
        skill = next(iter(competence.skills.values()))
        assert skill.steps[-1].action == "verify"
        assert any(s.action == "understand" for s in skill.steps)
        assert plan.task_context.schema.successes == 1

    def test_similar_goal_borrows_verified_prior(self):
        from frontal_lobe.reasoning import Planner

        competence = TaskCompetence()
        planner = Planner(_planning_network(), task_competence=competence)
        plan = planner.create_plan("fire", "understand")
        _run_plan(planner, plan)
        skill = next(iter(competence.skills.values()))

        # A structurally similar goal on an unknown concept: raw
        # feasibility is low, but the verified prior lifts it.
        cold = planner.create_plan("dragon", "understand")
        raw = planner._evaluate_feasibility(cold)
        assert cold.prior_skill_id == skill.skill_id
        assert cold.feasibility_score > raw

    def test_plan_skill_provenance_matches_weakest_step(self):
        """A consolidated plan claims only what its steps earned."""
        from frontal_lobe.reasoning import Planner

        # Steps marked by an epistemic verifier → "solver" skill.
        competence = TaskCompetence()
        planner = Planner(_planning_network(), task_competence=competence)
        plan = planner.create_plan("fire", "understand")
        while True:
            step = planner.advance(plan)
            if step is None:
                break
            planner.mark_step(
                plan,
                step,
                success=True,
                confidence=0.8,
                verification="solver",
            )
        skill = next(iter(competence.skills.values()))
        assert skill.verification == "solver"

        # Steps marked by bare assertion → "reported" skill.
        competence2 = TaskCompetence()
        planner2 = Planner(_planning_network(), task_competence=competence2)
        plan2 = planner2.create_plan("fire", "understand")
        _run_plan(planner2, plan2)
        skill2 = next(iter(competence2.skills.values()))
        assert skill2.verification == "reported"

        # A weaker-verified prior lifts feasibility less than a
        # solver-verified one on the same cold goal.
        cold = planner.create_plan("dragon", "understand")
        cold2 = planner2.create_plan("dragon", "understand")
        solver_lift = cold.feasibility_score - planner._evaluate_feasibility(cold)
        reported_lift = cold2.feasibility_score - planner2._evaluate_feasibility(cold2)
        assert solver_lift > reported_lift > 0.0

    def test_step_verification_survives_plan_persistence(self):
        from frontal_lobe.reasoning import Planner

        planner = Planner(_planning_network())
        plan = planner.create_plan("fire", "understand")
        step = planner.advance(plan)
        assert step is not None
        planner.mark_step(plan, step, success=True, verification="solver")

        restored = Planner(_planning_network())
        restored.restore_from_dict(planner.to_dict())
        rplan = next(p for p in restored._plans if p.goal == "fire")
        assert rplan.steps[0].verification == "solver"

    def test_blocked_plan_records_failure_without_skill(self):
        from frontal_lobe.reasoning import Planner, PlanStatus

        competence = TaskCompetence()
        net = ConceptNetwork()
        net.add_concept("fire")
        planner = Planner(net, task_competence=competence)

        plan = planner.create_plan("dragon", "understand")
        # "dragon" is unknown: its step can only fail, and with no
        # alternative path the plan blocks after max_attempts.
        _run_plan(planner, plan, succeed=False)

        assert plan.status == PlanStatus.BLOCKED
        assert plan.episode_recorded
        assert competence.skill_count == 0
        schema = plan.task_context.schema
        assert schema.episodes == 1
        assert schema.failures == 1


def _rule_context(
    competence: TaskCompetence,
    *,
    domain: str,
    feature_key: str,
    feature_value: bool,
    actions: tuple[str, ...],
) -> TaskContext:
    """A one-feature, two-action task — the probe family shape."""
    return competence.recognize(
        domain=domain,
        state={feature_key: feature_value},
        actions=actions,
        goal_conditions=[GoalCondition("solved", "eq", True)],
        roles=("inspect", "select", "support"),
    )


def _observe_rule_trial(
    competence: TaskCompetence,
    *,
    domain: str,
    feature_key: str,
    target: str,
    feature_value: bool,
    prefix: str,
    record_skill: bool = False,
) -> None:
    """Run one hidden-rule trial the way the probe harness does.

    The environment's rule: ``<prefix>`` role solves when the flag is
    set, ``<prefix>_false_`` role solves otherwise. The competence
    substrate sees only state transitions and success — no hint.
    """
    actions = (f"{prefix}_{target}", f"{prefix}_false_{target}")
    expected_role = prefix if feature_value else "false"
    context = _rule_context(
        competence,
        domain=domain,
        feature_key=feature_key,
        feature_value=feature_value,
        actions=actions,
    )
    for action in actions:
        observed_role = "false" if "_false_" in action else prefix
        after = {
            feature_key: feature_value,
            "solved": observed_role == expected_role,
        }
        competence.record_transition(
            context,
            action,
            {feature_key: feature_value},
            after,
            success=after["solved"],
        )
        if after["solved"]:
            if record_skill:
                competence.record_episode(
                    context,
                    steps=(
                        ProcedureStep(
                            action=action,
                            parameters={"support": 1.0},
                        ),
                    ),
                    success=True,
                    verification_score=1.0,
                    goal_conditions=[GoalCondition("solved", "eq", True)],
                    verification="external",
                )
            break


class TestTaskCompetenceGeneralization:
    """Abstraction and rebinding — the induced-rule layer."""

    def test_lifted_pattern_predicts_held_out_binding(self):
        """Training on align_ravel/align_tavi under flag-true yields a
        prediction for never-seen align_merek — the rule, not the name."""
        competence = TaskCompetence()
        _observe_rule_trial(
            competence,
            domain="lattice",
            feature_key="amber",
            target="ravel",
            feature_value=True,
            prefix="align",
        )
        _observe_rule_trial(
            competence,
            domain="lattice",
            feature_key="amber",
            target="sorn",
            feature_value=False,
            prefix="align",
        )
        _observe_rule_trial(
            competence,
            domain="lattice",
            feature_key="amber",
            target="tavi",
            feature_value=True,
            prefix="align",
        )

        held_out = _rule_context(
            competence,
            domain="lattice",
            feature_key="amber",
            feature_value=True,
            actions=("align_merek", "align_false_merek"),
        )
        predicted = competence.predict(held_out, "align_merek", {"amber": True})
        assert any(p.key == "solved" and p.expected is True for p in predicted)

    def test_conditional_model_abstains_on_unseen_shape(self):
        """An action only ever observed under flag-true yields no
        prediction under flag-false — abstention instead of averaging
        over the condition that decides the outcome."""
        competence = TaskCompetence()
        _observe_rule_trial(
            competence,
            domain="lattice",
            feature_key="amber",
            target="ravel",
            feature_value=True,
            prefix="align",
        )
        other_shape = _rule_context(
            competence,
            domain="lattice",
            feature_key="amber",
            feature_value=False,
            actions=("align_sorn", "align_false_sorn"),
        )
        # The schema merged (same signature shape modulo the flag
        # value); align_ravel was never observed under flag-false.
        predicted = competence.predict(other_shape, "align_ravel", {"amber": False})
        assert all(p.expected is not True for p in predicted)

    def test_foreign_skill_rebinds_by_step_position(self):
        """A lattice skill recalled in the archive domain maps its
        step position onto the target action order — 'the action at
        this position solved the analogous task'."""
        competence = TaskCompetence()
        _observe_rule_trial(
            competence,
            domain="lattice",
            feature_key="amber",
            target="ravel",
            feature_value=True,
            prefix="align",
            record_skill=True,
        )

        archive = _rule_context(
            competence,
            domain="archive",
            feature_key="ochre",
            feature_value=True,
            actions=("index", "discard"),
        )
        # The retrieval must actually surface the foreign skill.
        assert any(m.skill.signature.domain == "lattice" for m in archive.skills)
        predicted = competence.predict(archive, "index", {"ochre": True})
        assert any(p.key == "solved" and p.expected is True for p in predicted)
        # And the wrong position must not be predicted.
        wrong = competence.predict(archive, "discard", {"ochre": True})
        assert all(p.expected is not True for p in wrong)

    def test_rebinding_requires_matching_state_shape(self):
        """A foreign skill learned under flag-true says nothing about
        a flag-false situation — the condition is part of the rule."""
        competence = TaskCompetence()
        _observe_rule_trial(
            competence,
            domain="lattice",
            feature_key="amber",
            target="ravel",
            feature_value=True,
            prefix="align",
            record_skill=True,
        )
        archive_false = _rule_context(
            competence,
            domain="archive",
            feature_key="ochre",
            feature_value=False,
            actions=("index", "discard"),
        )
        for action in ("index", "discard"):
            predicted = competence.predict(archive_false, action, {"ochre": False})
            assert all(p.expected is not True for p in predicted)

    def test_abstraction_state_round_trips(self):
        """action_order, lifted models, conditional effects, and step
        positions all survive serialization."""
        competence = TaskCompetence()
        _observe_rule_trial(
            competence,
            domain="lattice",
            feature_key="amber",
            target="ravel",
            feature_value=True,
            prefix="align",
        )
        _observe_rule_trial(
            competence,
            domain="lattice",
            feature_key="amber",
            target="tavi",
            feature_value=True,
            prefix="align",
            record_skill=True,
        )

        restored = TaskCompetence.from_dict(competence.to_dict())
        held_out = _rule_context(
            restored,
            domain="lattice",
            feature_key="amber",
            feature_value=True,
            actions=("align_nema", "align_false_nema"),
        )
        predicted = restored.predict(held_out, "align_nema", {"amber": True})
        assert any(p.key == "solved" and p.expected is True for p in predicted)
        archive = _rule_context(
            restored,
            domain="archive",
            feature_key="ochre",
            feature_value=True,
            actions=("index", "discard"),
        )
        rebound = restored.predict(archive, "index", {"ochre": True})
        assert any(p.key == "solved" and p.expected is True for p in rebound)

    def test_exact_model_beats_foreign_rebinding(self):
        """Observed evidence for this action under this condition
        outranks an analogous foreign skill."""
        competence = TaskCompetence()
        _observe_rule_trial(
            competence,
            domain="lattice",
            feature_key="amber",
            target="ravel",
            feature_value=True,
            prefix="align",
            record_skill=True,
        )
        # Same domain, direct observation of the queried action under
        # the same shape — the foreign skill must not override it.
        context = _rule_context(
            competence,
            domain="lattice",
            feature_key="amber",
            feature_value=True,
            actions=("align_sorn", "align_false_sorn"),
        )
        # Two direct observations — the concrete model's reliability
        # (observations/3) then exceeds any rebound's halved match score.
        for _ in range(2):
            competence.record_transition(
                context,
                "align_sorn",
                {"amber": True},
                {"amber": True, "solved": True},
                success=True,
            )
        predicted = competence.predict(context, "align_sorn", {"amber": True})
        assert predicted[0].key == "solved"
        assert predicted[0].expected is True
        # Concrete evidence — not the weaker rebound confidence.
        assert predicted[0].confidence > 0.5
