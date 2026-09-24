"""Task competence — schema recognition, affordances, and skill reuse."""

from genesis_cognitive.concepts import ConceptNetwork
from genesis_cognitive.narrative import NarrativeEngine
from genesis_cognitive.persistence import (
    load_state,
    restore_task_competence,
    save_state,
)
from genesis_cognitive.reasoning import (
    GoalCondition,
    ProcedureStep,
    TaskCompetence,
    TaskContext,
)
from genesis_cognitive.self import ReflectionEngine, SelfModel
from genesis_cognitive.spatial import Grid, SpatialReasoner


def _grid(rows: list[list[int]]) -> Grid:
    return Grid.from_lists(rows)


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
            context, steps=steps, success=True, verification_score=1.0,
            verification="solver",
        )
        assert skill is not None
        assert skill.verification == "solver"

        # A weaker re-verification must not downgrade the record.
        same = competence.record_episode(
            context, steps=steps, success=True, verification_score=1.0,
            verification="reported",
        )
        assert same is skill
        assert skill.verification == "solver"

        # Stronger evidence upgrades it.
        competence.record_episode(
            context, steps=steps, success=True, verification_score=1.0,
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
                "self.r": 5, "self.c": 5,
                "target.distance": 8, "objects.count": 3,
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
                "self.r": 1, "self.c": 1,
                "target.distance": 5, "objects.count": 4,
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
            context, steps=[ProcedureStep("move", {"how": "thought"})],
            success=True, verification_score=1.0, verification="solver",
        )
        worldly = competence.record_episode(
            context, steps=[ProcedureStep("move", {"how": "acted"})],
            success=True, verification_score=1.0, verification="external",
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
        assert competence.verify(
            goals, {"target.distance": 0, "agent.alive": True}
        ) == 1.0
        assert competence.verify(
            goals, {"target.distance": 2, "agent.alive": True}
        ) < 1.0

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
        assert similar.skills[0].skill.steps == (step,)

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
            ReflectionEngine(network),
            NarrativeEngine(self_model, network),
            self_model,
            task_competence=competence,
        )
        data = load_state(str(tmp_path))
        assert data is not None
        restored = TaskCompetence()
        restore_task_competence(restored, data["task_competence"])
        assert restored.skill_count == 1


    def test_verified_skill_is_grounded_in_concept_network(self):
        from genesis_cognitive.concepts import ConceptNetwork, RelationType

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
        assert (
            net.get_concept("goal:target.distance|eq|0") is not None
        )


class TestSpatialCompetenceAdapter:
    def test_verified_spatial_rule_becomes_reusable_skill(self):
        competence = TaskCompetence()
        reasoner = SpatialReasoner(task_competence=competence)
        inp = _grid([[0, 1, 0], [0, 0, 0], [0, 0, 2]])
        out = _grid([[0, 0, 0], [0, 0, 0], [0, 1, 2]])
        test = _grid([[2, 0, 0], [0, 0, 0], [0, 0, 0]])

        sol = reasoner.solve([(inp, out)], [test], max_depth=1)
        assert sol.solved
        reasoner.record_task_outcome(
            sol, success=True, score=1.0, examples=[(inp, out)]
        )

        assert competence.skill_count == 1
        assert sol.task_context is not None
        assert sol.task_context.schema.successes == 1
        assert sol.task_context.schema.operators

    def test_skill_transfers_to_fresh_reasoner_on_similar_task(self):
        competence = TaskCompetence()
        first = SpatialReasoner(task_competence=competence)
        inp = _grid([[0, 1, 0], [0, 0, 0], [0, 0, 2]])
        out = _grid([[0, 0, 0], [0, 0, 0], [0, 1, 2]])
        sol = first.solve(
            [(inp, out)],
            [_grid([[0, 0, 0], [0, 0, 0], [0, 0, 0]])],
        )
        assert sol.solved
        first.record_task_outcome(
            sol, success=True, score=1.0, examples=[(inp, out)]
        )
        skill_id = next(iter(competence.skills))

        second = SpatialReasoner(task_competence=competence)
        inp2 = _grid([[0, 0, 0], [0, 3, 0], [0, 0, 0]])
        out2 = _grid([[0, 0, 0], [0, 0, 0], [0, 3, 0]])
        sol2 = second.solve(
            [(inp2, out2)],
            [_grid([[0, 4, 0], [0, 0, 0], [0, 0, 0]])],
            max_depth=1,
        )

        assert skill_id in sol2.retrieved_skills
        assert sol2.solved
        assert sol2.hypothesis is not None
        assert sol2.hypothesis.transforms[0].params["skill"] == skill_id
        assert sol2.predictions[0] == _grid(
            [[0, 0, 0], [0, 0, 0], [0, 4, 0]]
        )

    def test_retrieved_skills_compose_into_new_procedures(self):
        """Two verified skills chained solve a task neither solves alone."""
        competence = TaskCompetence()

        # Skill 1: gravity — a loose cell falls to the floor.
        first = SpatialReasoner(task_competence=competence)
        a_in = _grid([[0, 7, 0], [0, 0, 0], [0, 0, 2]])
        a_out = _grid([[0, 0, 0], [0, 0, 0], [0, 7, 2]])
        sol_a = first.solve([(a_in, a_out)], [_grid([[0] * 3] * 3)])
        assert sol_a.solved
        first.record_task_outcome(
            sol_a, success=True, score=1.0, examples=[(a_in, a_out)]
        )

        # Skill 2: fill an enclosed hole (background stays untouched,
        # so a plain recolor can't explain it).
        second = SpatialReasoner(task_competence=competence)
        b_in = _grid(
            [
                [0, 0, 0, 0, 0],
                [0, 2, 2, 2, 0],
                [0, 2, 0, 2, 0],
                [0, 2, 2, 2, 0],
                [0, 0, 0, 0, 0],
            ]
        )
        b_out = _grid(
            [
                [0, 0, 0, 0, 0],
                [0, 2, 2, 2, 0],
                [0, 2, 4, 2, 0],
                [0, 2, 2, 2, 0],
                [0, 0, 0, 0, 0],
            ]
        )
        sol_b = second.solve([(b_in, b_out)], [_grid([[0] * 5] * 5)])
        assert sol_b.solved
        second.record_task_outcome(
            sol_b, success=True, score=1.0, examples=[(b_in, b_out)]
        )
        assert competence.skill_count == 2

        # Composite task: drop the loose 7 AND fill the ring's hole.
        # Neither stored procedure covers the other; the only exact
        # explanations are compositions.
        third = SpatialReasoner(task_competence=competence)
        c_in = _grid(
            [
                [7, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 2, 2, 2],
                [0, 0, 0, 0, 2, 0, 2],
                [0, 0, 0, 0, 2, 2, 2],
            ]
        )
        c_out = _grid(
            [
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 2, 2, 2],
                [0, 0, 0, 0, 2, 4, 2],
                [7, 0, 0, 0, 2, 2, 2],
            ]
        )
        sol = third.solve([(c_in, c_out)], [_grid([[0] * 7] * 7)])
        assert sol.solved
        assert sol.hypothesis is not None
        used = [t.params.get("skill") for t in sol.hypothesis.transforms]
        assert len([s for s in used if s]) == 2
        assert set(used) == set(sol.retrieved_skills)

        # A failed outcome marks every skill in the composed sequence.
        third.record_task_outcome(
            sol, success=False, score=0.0, examples=[(c_in, c_out)]
        )
        for skill_id in sol.retrieved_skills:
            assert competence.skills[skill_id].failures == 1

    def test_primitive_repairs_retrieved_skill_near_miss(self):
        """A primitive before the skill fixes what the skill misses."""
        competence = TaskCompetence()
        first = SpatialReasoner(task_competence=competence)
        b_in = _grid(
            [
                [0, 0, 0, 0, 0],
                [0, 2, 2, 2, 0],
                [0, 2, 0, 2, 0],
                [0, 2, 2, 2, 0],
                [0, 0, 0, 0, 0],
            ]
        )
        b_out = _grid(
            [
                [0, 0, 0, 0, 0],
                [0, 2, 2, 2, 0],
                [0, 2, 4, 2, 0],
                [0, 2, 2, 2, 0],
                [0, 0, 0, 0, 0],
            ]
        )
        sol_b = first.solve([(b_in, b_out)], [_grid([[0] * 5] * 5)])
        assert sol_b.solved
        first.record_task_outcome(
            sol_b, success=True, score=1.0, examples=[(b_in, b_out)]
        )
        skill_id = next(iter(competence.skills))

        # Same ring-with-hole plus a stray 7 the skill can't remove:
        # the stored procedure alone is a near-miss that a primitive
        # must repair.
        second = SpatialReasoner(task_competence=competence)
        n_in = _grid(
            [
                [7, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 2, 2, 2, 0],
                [0, 0, 2, 0, 2, 0],
                [0, 0, 2, 2, 2, 0],
                [0, 0, 0, 0, 0, 0],
            ]
        )
        n_out = _grid(
            [
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 2, 2, 2, 0],
                [0, 0, 2, 4, 2, 0],
                [0, 0, 2, 2, 2, 0],
                [0, 0, 0, 0, 0, 0],
            ]
        )
        sol = second.solve([(n_in, n_out)], [_grid([[0] * 6] * 6)])
        assert sol.solved
        assert skill_id in sol.retrieved_skills
        assert sol.hypothesis is not None
        transforms = sol.hypothesis.transforms
        assert len(transforms) == 2
        # The skill macro lands after the repair primitive — a
        # composition impossible when skills were depth-zero only.
        assert "skill" not in transforms[0].params
        assert transforms[1].params["skill"] == skill_id

    def test_schema_learned_search_prior_accelerates_new_instance(self):
        """Winning families become the schema's search prior."""
        competence = TaskCompetence()
        first = SpatialReasoner(task_competence=competence)
        f_in = _grid(
            [
                [0, 0, 0, 0, 0],
                [0, 2, 2, 2, 0],
                [0, 2, 0, 2, 0],
                [0, 2, 2, 2, 0],
                [0, 0, 0, 0, 0],
            ]
        )
        f_out = _grid(
            [
                [0, 0, 0, 0, 0],
                [0, 2, 2, 2, 0],
                [0, 2, 4, 2, 0],
                [0, 2, 2, 2, 0],
                [0, 0, 0, 0, 0],
            ]
        )
        sol1 = first.solve([(f_in, f_out)], [_grid([[0] * 5] * 5)])
        assert sol1.solved
        first.record_task_outcome(
            sol1, success=True, score=1.0, examples=[(f_in, f_out)]
        )
        schema = sol1.task_context.schema
        assert schema.family_priors.get("fill", 0) > 0

        # New instance of the same family: different ring color,
        # different hole color — same structure.
        n_in = _grid(
            [
                [0, 0, 0, 0, 0, 0],
                [0, 3, 3, 3, 3, 0],
                [0, 3, 0, 0, 3, 0],
                [0, 3, 3, 3, 3, 0],
                [0, 0, 0, 0, 0, 0],
            ]
        )
        n_out = _grid(
            [
                [0, 0, 0, 0, 0, 0],
                [0, 3, 3, 3, 3, 0],
                [0, 3, 8, 8, 3, 0],
                [0, 3, 3, 3, 3, 0],
                [0, 0, 0, 0, 0, 0],
            ]
        )

        # Cold baseline: no priors at all.
        cold = SpatialReasoner(task_competence=TaskCompetence())
        cold_sol = cold.solve([(n_in, n_out)], [_grid([[0] * 6] * 6)])
        assert cold_sol.solved

        warm = SpatialReasoner(task_competence=competence)
        warm_sol = warm.solve([(n_in, n_out)], [_grid([[0] * 6] * 6)])
        assert warm_sol.solved
        # The learned prior put the fill family in the preferred
        # pass — the same task is solved in a fraction of the nodes.
        assert warm_sol.nodes_explored < cold_sol.nodes_explored

    def test_skill_rebinds_stored_params_to_new_evidence(self):
        """A stored procedure re-parameterizes: fill(4) → fill(8)."""
        competence = TaskCompetence()
        first = SpatialReasoner(task_competence=competence)
        f_in = _grid(
            [
                [0, 0, 0, 0, 0],
                [0, 2, 2, 2, 0],
                [0, 2, 0, 2, 0],
                [0, 2, 2, 2, 0],
                [0, 0, 0, 0, 0],
            ]
        )
        f_out = _grid(
            [
                [0, 0, 0, 0, 0],
                [0, 2, 2, 2, 0],
                [0, 2, 4, 2, 0],
                [0, 2, 2, 2, 0],
                [0, 0, 0, 0, 0],
            ]
        )
        sol1 = first.solve([(f_in, f_out)], [_grid([[0] * 5] * 5)])
        assert sol1.solved
        first.record_task_outcome(
            sol1, success=True, score=1.0, examples=[(f_in, f_out)]
        )

        # New instance needs a different fill color — the stored
        # binding (4) is a binding, not the procedure.
        second = SpatialReasoner(task_competence=competence)
        n_in = _grid(
            [
                [0, 0, 0, 0, 0],
                [0, 3, 3, 3, 0],
                [0, 3, 0, 3, 0],
                [0, 3, 3, 3, 0],
                [0, 0, 0, 0, 0],
            ]
        )
        n_out = _grid(
            [
                [0, 0, 0, 0, 0],
                [0, 3, 3, 3, 0],
                [0, 3, 8, 3, 0],
                [0, 3, 3, 3, 0],
                [0, 0, 0, 0, 0],
            ]
        )
        test_in = _grid(
            [
                [0, 0, 0, 0, 0],
                [0, 5, 5, 5, 0],
                [0, 5, 0, 5, 0],
                [0, 5, 5, 5, 0],
                [0, 0, 0, 0, 0],
            ]
        )
        second._recognize_task([(n_in, n_out)], [test_in])
        macros = second._skill_transforms([(n_in, n_out)])
        rebound = [t for t in macros if "rebound" in t.params]
        assert rebound, "no re-parameterized skill variant proposed"
        # The rebound variant applies the stored procedure with the
        # new task's evidence — fill(8), not the recorded fill(4).
        assert rebound[0](n_in) == n_out
        assert rebound[0](test_in) == _grid(
            [
                [0, 0, 0, 0, 0],
                [0, 5, 5, 5, 0],
                [0, 5, 8, 5, 0],
                [0, 5, 5, 5, 0],
                [0, 0, 0, 0, 0],
            ]
        )
        sol = second.solve([(n_in, n_out)], [test_in])
        assert sol.solved
        assert sol.predictions[0] == rebound[0](test_in)

    def test_math_episodes_consolidate_solver_verified_skills(self):
        """try_math records episodes; '2+3' and '5+7' share a schema."""
        from genesis_cognitive.reasoning.math_reasoning import try_math

        competence = TaskCompetence()
        r1 = try_math("what is 2 + 3", task_competence=competence)
        assert r1 is not None and r1.answer == "5"
        try_math("what is 5 + 7", task_competence=competence)
        try_math("solve 2x = 10", task_competence=competence)

        total_episodes = sum(s.episodes for s in competence.schemas.values())
        assert total_episodes == 3
        # A computation skill consolidated with honest provenance —
        # the engine's own evaluation, not an external claim.
        compute_skills = [
            s
            for s in competence.skills.values()
            if s.verification == "solver"
            and any(st.family == "computation" for st in s.steps)
        ]
        assert compute_skills
        # The solve path recorded its stepwise procedure.
        assert any(
            st.family == "solution"
            for s in competence.skills.values()
            for st in s.steps
        )

    def test_math_failure_marks_schema_not_skill(self):
        from genesis_cognitive.reasoning.math_reasoning import try_math

        competence = TaskCompetence()
        result = try_math(
            "prove that 2 + 2 = 5", task_competence=competence
        )
        assert result is not None and not result.success
        assert sum(s.failures for s in competence.schemas.values()) == 1
        # A disproven proof consolidates nothing.
        assert not any(
            st.family == "proof"
            for s in competence.skills.values()
            for st in s.steps
        )

    def test_non_math_question_records_no_episode(self):
        from genesis_cognitive.reasoning.math_reasoning import try_math

        competence = TaskCompetence()
        assert try_math("hello there friend", task_competence=competence) is None
        assert not competence.schemas

    def test_low_salience_episode_stays_episodic_not_procedural(self):
        """Neuromodulatory gate: verified but low-salience episodes
        enter the schema ledger without consolidating a skill."""
        competence = TaskCompetence()
        context = _context(competence)
        steps = [ProcedureStep("move", {"to": "target"})]

        skill = competence.record_episode(
            context,
            steps=steps,
            success=True,
            verification_score=1.0,
            verification="external",
            salience=0.0,
        )
        assert skill is None
        # The episode is remembered — episodic ledger records it —
        # but nothing proceduralized.
        assert context.schema.episodes == 1
        assert context.schema.successes == 1
        assert not competence.skills

        # The same verified outcome under high salience consolidates.
        skill = competence.record_episode(
            context,
            steps=steps,
            success=True,
            verification_score=1.0,
            verification="external",
            salience=0.9,
        )
        assert skill is not None
        assert context.schema.episodes == 2

    def test_ambient_salience_broadcast_gates_consolidation(self):
        """A salience_getter on the substrate modulates every
        consolidation without the caller threading it through."""
        gain = {"level": 0.0}
        competence = TaskCompetence(
            salience_getter=lambda: gain["level"]
        )
        context = _context(competence)
        steps = [ProcedureStep("move")]

        # Low ambient gain: verified episode stays episodic.
        assert (
            competence.record_episode(
                context,
                steps=steps,
                success=True,
                verification_score=1.0,
                verification="external",
            )
            is None
        )
        assert not competence.skills

        # Raise the ambient level: the same episode proceduralizes.
        gain["level"] = 0.9
        assert (
            competence.record_episode(
                context,
                steps=steps,
                success=True,
                verification_score=1.0,
                verification="external",
            )
            is not None
        )

    def test_skill_macros_proposed_beyond_first_step(self):
        """_expand_hypothesis offers skills to nonempty hypotheses."""
        from genesis_cognitive.spatial.solver import SpatialHypothesis

        competence = TaskCompetence()
        first = SpatialReasoner(task_competence=competence)
        a_in = _grid([[0, 7, 0], [0, 0, 0], [0, 0, 2]])
        a_out = _grid([[0, 0, 0], [0, 0, 0], [0, 7, 2]])
        sol_a = first.solve([(a_in, a_out)], [_grid([[0] * 3] * 3)])
        assert sol_a.solved
        first.record_task_outcome(
            sol_a, success=True, score=1.0, examples=[(a_in, a_out)]
        )

        second = SpatialReasoner(task_competence=competence)
        b_in = _grid([[0, 3, 0], [0, 0, 0], [0, 0, 5]])
        b_out = _grid([[0, 0, 0], [0, 0, 0], [0, 3, 5]])
        second._recognize_task([(b_in, b_out)], [_grid([[0] * 3] * 3)])
        macros = second._skill_transforms()
        assert macros

        # Expand a hypothesis that already contains a transform:
        # skill macros must still appear among the proposals.
        hyp = SpatialHypothesis([macros[0]], 0.0)
        candidates, _, _ = second._expand_hypothesis(
            hyp,
            [(b_in, b_out)],
            True,
            lambda seq: (0.0, 0),
            0,
            float("inf"),
            10_000,
        )
        assert any(
            len(h.transforms) == 2 and "skill" in h.transforms[1].params
            for h in candidates
        )


class _NavEnv:
    """Minimal navigation world: one avatar cell, one goal cell."""

    def __init__(
        self,
        *,
        avatar_color: int = 3,
        goal_color: int = 7,
        size: int = 11,
        start: tuple[int, int] = (5, 5),
        goal: tuple[int, int] = (1, 1),
    ) -> None:
        self.size = size
        self.avatar_color = avatar_color
        self.goal_color = goal_color
        self.pos = list(start)
        self.goal = goal
        self.won = False
        self.actions = ["up", "down", "left", "right"]

    def frame(self) -> list[list[int]]:
        g = [[0] * self.size for _ in range(self.size)]
        if not self.won:
            g[self.goal[0]][self.goal[1]] = self.goal_color
        g[self.pos[0]][self.pos[1]] = self.avatar_color
        return g

    def step(self, action: str) -> tuple[list[list[int]], str]:
        if action == "up":
            self.pos[0] = max(0, self.pos[0] - 1)
        elif action == "down":
            self.pos[0] = min(self.size - 1, self.pos[0] + 1)
        elif action == "left":
            self.pos[1] = max(0, self.pos[1] - 1)
        elif action == "right":
            self.pos[1] = min(self.size - 1, self.pos[1] + 1)
        if tuple(self.pos) == self.goal:
            self.won = True
        return self.frame(), "WIN" if self.won else "PLAYING"


class TestAgentCompetenceAdapter:
    def test_steps_feed_shared_transition_model(self):
        from genesis_cognitive.spatial import SpatialAgent, frame_to_grid

        competence = TaskCompetence()
        env = _NavEnv()
        agent = SpatialAgent(seed=0, epsilon=1.0, task_competence=competence)
        agent.observe(env.frame())
        first = agent.choose_action(env.actions)  # recognizes the env
        env.step(first)
        agent.observe(env.frame())

        # Scripted moves: "right" twice teaches a +1 column delta.
        for action in ("right", "right"):
            agent._last_action = action
            env.step(action)
            agent.observe(env.frame())

        context = agent._task_context
        assert context is not None
        assert competence.transition_count >= 3
        predicted = competence.predict(
            context,
            "right",
            agent._transition_state(frame_to_grid(env.frame())),
        )
        by_key = {p.key: p for p in predicted}
        assert by_key["avatar.c"].delta == 1.0

    def test_win_consolidates_control_map_for_transfer(self):
        from genesis_cognitive.spatial import SpatialAgent

        competence = TaskCompetence()
        env = _NavEnv()
        agent = SpatialAgent(seed=0, epsilon=0.0, task_competence=competence)
        agent.observe(env.frame())
        # Control model learned this episode (as navigation would).
        for a, (dr, dc) in {
            "up": (-1, 0), "down": (1, 0),
            "left": (0, -1), "right": (0, 1),
        }.items():
            st = agent._stat(a)
            st.attempts, st.dr, st.dc = 1, dr, dc
        agent.avatar_color = 3

        state = "PLAYING"
        for _ in range(20):
            action = agent.choose_action(env.actions)
            frame, state = env.step(action)
            agent.observe(frame)
            agent.mark_goal_reached()
            if state == "WIN":
                break
        assert state == "WIN"
        agent.on_episode_end("WIN")

        assert competence.skill_count == 1
        skill = next(iter(competence.skills.values()))
        control = {s.action for s in skill.steps if s.family == "control"}
        assert control == {"up", "down", "left", "right"}
        # Reaching the goal consumed it — the pickup role was learned.
        assert any(
            s.parameters.get("role") == "pickup" for s in skill.steps
        )

        # A fresh agent on a similar world inherits the control map.
        env2 = _NavEnv(goal_color=5, goal=(1, 9))
        agent2 = SpatialAgent(
            seed=1, epsilon=0.0, task_competence=competence
        )
        agent2.observe(env2.frame())
        agent2.avatar_color = 3
        action = agent2.choose_action(env2.actions)

        assert agent2._skill_priors["up"] == (-1.0, 0.0)
        assert agent2._skill_priors["right"] == (0.0, 1.0)
        # Goal (1,9) from (5,5): up or right both close distance;
        # either proves prior-based navigation without exploration.
        assert action in ("up", "right")
        assert all(s.attempts == 0 for s in agent2.stats.values())

    def test_game_over_demotes_adopted_skill(self):
        from genesis_cognitive.spatial import SpatialAgent

        competence = TaskCompetence()
        env = _NavEnv()
        agent = SpatialAgent(seed=0, epsilon=0.0, task_competence=competence)
        agent.observe(env.frame())
        for a, (dr, dc) in {
            "up": (-1, 0), "down": (1, 0),
            "left": (0, -1), "right": (0, 1),
        }.items():
            st = agent._stat(a)
            st.attempts, st.dr, st.dc = 1, dr, dc
        agent.avatar_color = 3
        agent.choose_action(env.actions)
        agent.on_episode_end("WIN")
        skill = next(iter(competence.skills.values()))
        wins = skill.successes

        env2 = _NavEnv(goal_color=5, goal=(1, 9))
        agent2 = SpatialAgent(
            seed=1, epsilon=0.0, task_competence=competence
        )
        agent2.observe(env2.frame())
        agent2.avatar_color = 3
        agent2.choose_action(env2.actions)
        assert skill.skill_id in agent2._episode_skill_ids
        agent2.on_episode_end("GAME_OVER")

        assert skill.failures == 1
        assert skill.successes == wins
        assert agent2._task_context is None

    def test_surprise_renews_exploration(self):
        """A transition that defies the learned model raises epsilon."""
        from genesis_cognitive.spatial import SpatialAgent

        competence = TaskCompetence()
        env = _NavEnv(goal=(3, 5))
        agent = SpatialAgent(seed=0, epsilon=0.0, task_competence=competence)
        agent.observe(env.frame())
        agent.choose_action(env.actions)  # recognize the env

        # Build a reliable "up" model: avatar.r -1 per step.
        agent._last_action = "up"
        env.step("up")
        agent.observe(env.frame())
        assert agent._grit_steps == 0

        # The next "up" reaches the goal, which vanishes — object and
        # cell counts change without any learned precedent.
        agent._last_action = "up"
        env.step("up")
        agent.observe(env.frame())
        assert agent._grit_steps > 0

    def test_hazard_role_prior_lowers_evidence_bar(self):
        """A won env with a mover teaches 'hazard' as a role prior."""
        from genesis_cognitive.spatial import SpatialAgent

        competence = TaskCompetence()
        env = _NavEnv()
        agent = SpatialAgent(seed=0, epsilon=0.0, task_competence=competence)
        agent.observe(env.frame())
        for a, (dr, dc) in {
            "up": (-1, 0), "down": (1, 0),
            "left": (0, -1), "right": (0, 1),
        }.items():
            st = agent._stat(a)
            st.attempts, st.dr, st.dc = 1, dr, dc
        agent.avatar_color = 3
        # It empirically learned this env family has self-moving
        # hazards before winning.
        agent._hazard_colors.add(9)
        agent.choose_action(env.actions)
        agent.on_episode_end("WIN")

        env2 = _NavEnv(goal_color=5, goal=(1, 9))
        agent2 = SpatialAgent(
            seed=1, epsilon=0.0, task_competence=competence
        )
        agent2.observe(env2.frame())
        agent2.avatar_color = 3
        agent2.choose_action(env2.actions)
        assert "hazard" in agent2._role_priors

        def mover_frame(apos, mpos):
            g = [[0] * 11 for _ in range(11)]
            g[apos[0]][apos[1]] = 3   # avatar
            g[mpos[0]][mpos[1]] = 9   # self-moving object
            return g

        # Avatar shifts 2 cells, mover shifts 1 — the mover can't be
        # mistaken for the avatar. A color must appear in both frames
        # of a pair to earn a vote, so three frames yield two votes.
        agent2._last_action = "right"
        agent2.observe(mover_frame((5, 7), (2, 2)))
        agent2._last_action = "right"
        agent2.observe(mover_frame((5, 9), (3, 2)))
        assert 9 not in agent2._hazard_colors
        agent2._last_action = "right"
        agent2.observe(mover_frame((5, 10), (4, 2)))
        assert 9 in agent2._hazard_colors

    def test_teleporting_object_earns_no_mover_votes(self):
        """A color reappearing elsewhere is a new object, not a mover.

        Identity binding: only a track with centroid continuity
        accrues motion evidence — a teleporting spawn can never fake
        a trajectory, no matter how often the color "moves".
        """
        from genesis_cognitive.spatial import SpatialAgent

        def frame(apos, mpos):
            g = [[0] * 11 for _ in range(11)]
            g[apos[0]][apos[1]] = 3   # avatar
            g[mpos[0]][mpos[1]] = 9   # object that keeps teleporting
            return g

        agent = SpatialAgent(seed=0, epsilon=0.0)
        agent.observe(frame((5, 4), (2, 2)))
        agent.avatar_color = 3
        # Avatar moves ≤3 cells per frame; the color jumps 6+ cells —
        # no track can bind it, so it can never accrue mover votes.
        for apos, mpos in (
            ((5, 7), (8, 8)),
            ((5, 10), (1, 9)),
            ((5, 8), (9, 1)),
            ((5, 5), (4, 6)),
        ):
            agent._last_action = "right"
            agent.observe(frame(apos, mpos))

        assert 9 not in agent._hazard_colors
        assert all(
            t.moved_frames == 0 for t in agent._tracks.values()
        )
        # The avatar was still found by its continuous displacement.
        assert agent._avatar_votes.get(3, 0) >= 3

    def test_hazard_speed_prior_marks_first_sighting(self):
        """Hazard dynamics transfer: a prior carrying the family's
        mover speed lets one matching observation suffice."""
        from genesis_cognitive.spatial import SpatialAgent

        competence = TaskCompetence()
        env = _NavEnv()
        agent = SpatialAgent(seed=0, epsilon=0.0, task_competence=competence)
        agent.observe(env.frame())
        for a, (dr, dc) in {
            "up": (-1, 0), "down": (1, 0),
            "left": (0, -1), "right": (0, 1),
        }.items():
            st = agent._stat(a)
            st.attempts, st.dr, st.dc = 1, dr, dc
        agent.avatar_color = 3
        # Movers in this family travel one cell per step.
        agent._hazard_colors.add(9)
        agent._hazard_velocity[9] = (1.0, 0.0)
        agent.choose_action(env.actions)
        agent.on_episode_end("WIN")

        skill = next(iter(competence.skills.values()))
        hazard = next(
            s for s in skill.steps
            if s.parameters.get("role") == "hazard"
        )
        assert hazard.parameters["speed"] == 1.0

        env2 = _NavEnv(goal_color=5, goal=(1, 9))
        agent2 = SpatialAgent(
            seed=1, epsilon=0.0, task_competence=competence
        )
        agent2.observe(env2.frame())
        agent2.avatar_color = 3
        agent2.choose_action(env2.actions)
        assert agent2._prior_hazard_speed == 1.0

        def frame(apos, mpos):
            g = [[0] * 11 for _ in range(11)]
            g[1][9] = 5               # goal persists
            g[apos[0]][apos[1]] = 3   # avatar
            g[mpos[0]][mpos[1]] = 9   # mover, speed 1
            return g

        # One sighting of motion matching the family speed marks it.
        agent2._last_action = "right"
        agent2.observe(frame((5, 7), (2, 2)))
        agent2._last_action = "right"
        agent2.observe(frame((5, 10), (3, 2)))
        assert 9 in agent2._hazard_colors

    def test_schema_model_guides_fresh_agent_without_skill(self):
        """Failed exploration still transfers: the schema's transition
        model carries the control map even when no skill consolidated.
        """
        from genesis_cognitive.spatial import SpatialAgent

        competence = TaskCompetence()
        env = _NavEnv()
        agent = SpatialAgent(seed=0, epsilon=1.0, task_competence=competence)
        agent.observe(env.frame())
        agent.choose_action(env.actions)
        # Explored, learned the controls, never won — no skill.
        for action in ("up", "right", "up", "right", "left", "down"):
            agent._last_action = action
            env.step(action)
            agent.observe(env.frame())
        assert competence.skill_count == 0
        assert competence.transition_count == 6

        env2 = _NavEnv(goal_color=5, goal=(1, 9))
        agent2 = SpatialAgent(
            seed=1, epsilon=0.0, task_competence=competence
        )
        agent2.observe(env2.frame())
        agent2.avatar_color = 3
        action = agent2.choose_action(env2.actions)

        # No local attempts, no skill — yet the schema model predicted
        # each action's displacement well enough to navigate.
        assert all(s.attempts == 0 for s in agent2.stats.values())
        assert not agent2._skill_priors
        assert action in ("up", "right")  # both close distance to (1,9)


def _planning_network() -> ConceptNetwork:
    """fire depends on oxygen and fuel; oxygen causes circulation."""
    from genesis_cognitive.concepts import RelationType

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
        from genesis_cognitive.reasoning import PlanningEngine

        competence = TaskCompetence()
        planner = PlanningEngine(
            _planning_network(), task_competence=competence
        )
        plan = planner.create_plan("fire", "understand")
        step = planner.advance(plan)
        assert step is not None
        planner.mark_step(plan, step, success=True, confidence=0.8)

        assert competence.transition_count == 1
        op = plan.task_context.schema.operators[step.step_type]
        assert op.successes == 1
        assert op.effects["step.status"].changed == 1

    def test_completed_plan_consolidates_procedure(self):
        from genesis_cognitive.reasoning import PlanningEngine, PlanStatus

        competence = TaskCompetence()
        planner = PlanningEngine(
            _planning_network(), task_competence=competence
        )
        plan = planner.create_plan("fire", "understand")
        _run_plan(planner, plan)

        assert plan.status == PlanStatus.COMPLETED
        assert competence.skill_count == 1
        skill = next(iter(competence.skills.values()))
        assert skill.steps[-1].action == "verify"
        assert any(s.action == "understand" for s in skill.steps)
        assert plan.task_context.schema.successes == 1

    def test_similar_goal_borrows_verified_prior(self):
        from genesis_cognitive.reasoning import PlanningEngine

        competence = TaskCompetence()
        planner = PlanningEngine(
            _planning_network(), task_competence=competence
        )
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
        from genesis_cognitive.reasoning import PlanningEngine

        # Steps marked by an epistemic verifier → "solver" skill.
        competence = TaskCompetence()
        planner = PlanningEngine(
            _planning_network(), task_competence=competence
        )
        plan = planner.create_plan("fire", "understand")
        while True:
            step = planner.advance(plan)
            if step is None:
                break
            planner.mark_step(
                plan, step, success=True, confidence=0.8,
                verification="solver",
            )
        skill = next(iter(competence.skills.values()))
        assert skill.verification == "solver"

        # Steps marked by bare assertion → "reported" skill.
        competence2 = TaskCompetence()
        planner2 = PlanningEngine(
            _planning_network(), task_competence=competence2
        )
        plan2 = planner2.create_plan("fire", "understand")
        _run_plan(planner2, plan2)
        skill2 = next(iter(competence2.skills.values()))
        assert skill2.verification == "reported"

        # A weaker-verified prior lifts feasibility less than a
        # solver-verified one on the same cold goal.
        cold = planner.create_plan("dragon", "understand")
        cold2 = planner2.create_plan("dragon", "understand")
        solver_lift = (
            cold.feasibility_score - planner._evaluate_feasibility(cold)
        )
        reported_lift = (
            cold2.feasibility_score - planner2._evaluate_feasibility(cold2)
        )
        assert solver_lift > reported_lift > 0.0

    def test_step_verification_survives_plan_persistence(self):
        from genesis_cognitive.reasoning import PlanningEngine

        planner = PlanningEngine(_planning_network())
        plan = planner.create_plan("fire", "understand")
        step = planner.advance(plan)
        assert step is not None
        planner.mark_step(
            plan, step, success=True, verification="solver"
        )

        restored = PlanningEngine(_planning_network())
        restored.restore_from_dict(planner.to_dict())
        rplan = next(p for p in restored._plans if p.goal == "fire")
        assert rplan.steps[0].verification == "solver"

    def test_blocked_plan_records_failure_without_skill(self):
        from genesis_cognitive.reasoning import PlanningEngine, PlanStatus

        competence = TaskCompetence()
        net = ConceptNetwork()
        net.add_concept("fire")
        planner = PlanningEngine(net, task_competence=competence)

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


class TestAssemblyCompetenceAdapter:
    """A fourth domain on the same substrate: edge-matching puzzles."""

    @staticmethod
    def _puzzle(rows: int, cols: int, seed: int):
        import random

        from genesis_cognitive.assembly import PiecePuzzle

        return PiecePuzzle.generate(rows, cols, random.Random(seed))

    def test_agent_solves_puzzle_and_consolidates_skill(self):
        from genesis_cognitive.assembly import AssemblyAgent

        competence = TaskCompetence()
        agent = AssemblyAgent(seed=1, task_competence=competence)
        result = agent.solve(self._puzzle(2, 3, 10), max_steps=800)

        assert result.solved
        assert competence.skill_count == 1
        skill = next(iter(competence.skills.values()))
        # The board's own constraint check is world-state evidence.
        assert skill.verification == "external"
        # The learned procedure is the fit→cost affordance map, not a
        # recorded answer key — no piece ids or slots in the params.
        placement = [
            s for s in skill.steps if s.family == "placement"
        ]
        assert placement
        assert all(
            set(s.parameters) == {"fits", "mean_cost"}
            for s in placement
        )

    def test_learned_affordance_orders_constraint_satisfaction(self):
        """The discovered rule: more satisfied edges → lower cost."""
        from genesis_cognitive.assembly import AssemblyAgent

        competence = TaskCompetence()
        agent = AssemblyAgent(seed=1, task_competence=competence)
        result = agent.solve(self._puzzle(2, 3, 10), max_steps=800)
        assert result.solved

        skill = next(iter(competence.skills.values()))
        costs = {
            s.parameters["fits"]: s.parameters["mean_cost"]
            for s in skill.steps
            if s.family == "placement"
        }
        assert max(costs) >= 3  # best placements exist
        assert costs[max(costs)] == 0.0
        assert min(costs) < max(costs)
        assert costs[min(costs)] > 0.0

    def test_skill_prior_transfers_to_new_instance(self):
        from genesis_cognitive.assembly import AssemblyAgent

        competence = TaskCompetence()
        first = AssemblyAgent(seed=1, task_competence=competence)
        cold = first.solve(self._puzzle(2, 3, 10), max_steps=800)
        assert cold.solved
        n_schemas = len(competence.schemas)

        second = AssemblyAgent(seed=2, task_competence=competence)
        warm = second.solve(self._puzzle(2, 3, 99), max_steps=800)

        assert warm.solved
        # Same family → same schema — the transfer was structural,
        # not a coincidence of identical piece codes.
        assert len(competence.schemas) == n_schemas
        assert second._fit_priors  # adopted the fit→cost prior
        # And it materially helped: the warm agent solves in far
        # fewer steps than cold trial-and-error.
        assert warm.steps < cold.steps

    def test_unsolvable_puzzle_marks_adopted_skill_failed(self):
        from genesis_cognitive.assembly import AssemblyAgent, Piece

        competence = TaskCompetence()
        first = AssemblyAgent(seed=1, task_competence=competence)
        assert first.solve(self._puzzle(2, 3, 10), max_steps=800).solved
        skill_id = next(iter(competence.skills))

        # Corrupt one piece: edge code 9 matches nothing and can't
        # face a border — the puzzle can never reach zero mismatches.
        broken = self._puzzle(2, 3, 20)
        pid = next(iter(broken.pool))
        broken.pool[pid] = Piece(pid, (9, 9, 9, 9))

        second = AssemblyAgent(seed=3, task_competence=competence)
        result = second.solve(broken, max_steps=200)
        assert not result.solved
        assert competence.skills[skill_id].failures == 1
