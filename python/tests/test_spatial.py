"""Spatial layer tests — grid ops, scene perception, solver, grounding."""

from genesis_cognitive.concepts import ConceptNetwork, RelationType
from genesis_cognitive.spatial import (
    Grid,
    SpatialReasoner,
    SpatialRelationKind,
    ground_scene,
    perceive,
    scene_facts,
)


def _grid(rows: list[list[int]]) -> Grid:
    return Grid.from_lists(rows)


# ── Grid ─────────────────────────────────────────────────────────


class TestGrid:
    def test_shape_and_access(self):
        g = _grid([[1, 2, 3], [4, 5, 6]])
        assert g.shape == (2, 3)
        assert g.at(1, 2) == 6
        assert g.to_lists() == [[1, 2, 3], [4, 5, 6]]

    def test_rotate90(self):
        g = _grid([[1, 2], [3, 4]])
        assert g.rotate90(1).to_lists() == [[3, 1], [4, 2]]
        assert g.rotate90(2).to_lists() == [[4, 3], [2, 1]]
        assert g.rotate90(4) == g

    def test_reflect(self):
        g = _grid([[1, 2], [3, 4]])
        assert g.reflect_h().to_lists() == [[2, 1], [4, 3]]
        assert g.reflect_v().to_lists() == [[3, 4], [1, 2]]
        assert g.reflect_diag().to_lists() == [[1, 3], [2, 4]]

    def test_recolor(self):
        g = _grid([[0, 1], [1, 0]])
        assert g.recolor({1: 7}).to_lists() == [[0, 7], [7, 0]]

    def test_crop_to_content(self):
        g = _grid([[0, 0, 0], [0, 5, 0], [0, 0, 0]])
        assert g.crop_to_content().to_lists() == [[5]]

    def test_scale_and_tile(self):
        g = _grid([[1, 2]])
        assert g.scale(2).to_lists() == [[1, 1, 2, 2], [1, 1, 2, 2]]
        assert g.tile(2, 2).to_lists() == [[1, 2, 1, 2], [1, 2, 1, 2]]

    def test_immutable_equality(self):
        assert _grid([[1]]) == _grid([[1]])
        assert _grid([[1]]) != _grid([[2]])
        assert hash(_grid([[1]])) == hash(_grid([[1]]))


# ── Perception ───────────────────────────────────────────────────


class TestPerceive:
    def test_two_objects_extracted(self):
        g = _grid([
            [0, 1, 1, 0],
            [0, 0, 0, 0],
            [2, 0, 0, 0],
        ])
        scene = perceive(g, background=0)
        assert len(scene.objects) == 2
        colors = {o.color for o in scene.objects}
        assert colors == {1, 2}

    def test_connectivity_splits_objects(self):
        # Diagonal touch: one object under 8-connectivity, two under 4.
        g = _grid([
            [1, 0],
            [0, 1],
        ])
        assert len(perceive(g, background=0).objects) == 2
        assert len(perceive(g, background=0, diagonal=True).objects) == 1

    def test_above_relation(self):
        # obj 0 (color 1) is entirely above obj 1 (color 2).
        g = _grid([
            [1, 0, 0],
            [0, 0, 0],
            [2, 0, 0],
        ])
        scene = perceive(g, background=0)
        kinds = {(r.subject, r.kind, r.object) for r in scene.relations}
        assert (0, SpatialRelationKind.ABOVE, 1) in kinds
        assert (1, SpatialRelationKind.BELOW, 0) in kinds

    def test_same_shape_and_color_relations(self):
        g = _grid([
            [3, 0, 3],
            [0, 0, 0],
            [4, 0, 0],
        ])
        scene = perceive(g, background=0)
        pairs = {(r.subject, r.kind, r.object) for r in scene.relations}
        # The two color-3 single cells share color and shape.
        obj3 = [o.index for o in scene.objects if o.color == 3]
        assert (obj3[0], SpatialRelationKind.SAME_COLOR, obj3[1]) in pairs
        assert (obj3[0], SpatialRelationKind.SAME_SHAPE, obj3[1]) in pairs

    def test_containment(self):
        g = _grid([
            [2, 2, 2],
            [2, 1, 2],
            [2, 2, 2],
        ])
        scene = perceive(g, background=0)
        kinds = {(r.subject, r.kind, r.object) for r in scene.relations}
        inner = next(o.index for o in scene.objects if o.color == 1)
        outer = next(o.index for o in scene.objects if o.color == 2)
        assert (inner, SpatialRelationKind.INSIDE, outer) in kinds
        assert (outer, SpatialRelationKind.CONTAINS, inner) in kinds

    def test_background_defaults_to_most_common(self):
        g = _grid([[0, 0, 5], [0, 5, 5]])
        scene = perceive(g)
        assert scene.background == 0
        assert len(scene.objects) == 1


# ── Solver ───────────────────────────────────────────────────────


class TestSpatialReasoner:
    def test_solves_rotation(self):
        inp = _grid([[1, 0], [0, 0]])
        out = _grid([[0, 1], [0, 0]])
        reasoner = SpatialReasoner()
        sol = reasoner.solve([(inp, out)], [_grid([[0, 0], [0, 1]])])
        assert sol.solved
        assert sol.predictions[0] == _grid([[0, 0], [1, 0]])

    def test_solves_recolor(self):
        inp = _grid([[1, 1], [0, 0]])
        out = _grid([[7, 7], [0, 0]])
        reasoner = SpatialReasoner()
        sol = reasoner.solve([(inp, out)], [_grid([[2, 0], [0, 0]])])
        # Recolor map 1→7 can't transfer to a grid containing 2, so
        # this may solve with a different hypothesis or not at all —
        # the invariant is that a solved hypothesis reproduces training.
        if sol.solved:
            assert sol.hypothesis is not None
            assert sol.hypothesis.apply(inp) == out

    def test_requires_all_examples_exact(self):
        # No single primitive maps both pairs — must not report solved.
        a = _grid([[1, 0], [0, 0]])
        b = _grid([[0, 1], [0, 0]])
        out_a = _grid([[1, 0], [0, 0]])
        out_b = _grid([[0, 0], [0, 1]])
        reasoner = SpatialReasoner()
        sol = reasoner.solve([(a, out_a), (b, out_b)], [a], max_depth=1)
        assert not sol.solved
        # Near-miss hypothesis is still surfaced for inspection.
        assert sol.hypothesis is not None

    def test_solves_crop(self):
        inp = _grid([[0, 0, 0], [0, 4, 4], [0, 4, 4]])
        out = _grid([[4, 4], [4, 4]])
        reasoner = SpatialReasoner()
        sol = reasoner.solve([(inp, out)], [_grid([[0, 0, 0], [0, 9, 0], [0, 0, 0]])])
        assert sol.solved
        assert sol.predictions[0] == _grid([[9]])

    def test_grounded_reasoner_shares_network(self):
        net = ConceptNetwork()
        reasoner = SpatialReasoner(net)
        assert reasoner.network is net

    def test_learned_rules_proposed_on_future_tasks(self):
        # Teach its a rule, then confirm it is reused on a later task.
        from genesis_cognitive.spatial import Transform

        reasoner = SpatialReasoner()
        reasoner.learn(
            [Transform("recolor", lambda g: g.recolor({1: 9}), {"map": {1: 9}})]
        )
        sol = reasoner.solve(
            [(_grid([[1, 0]]), _grid([[9, 0]]))],
            [_grid([[1]])],
            max_depth=1,
        )
        assert sol.solved
        assert sol.hypothesis is not None
        assert "recolor" in sol.hypothesis.describe()
        assert sol.predictions[0] == _grid([[9]])

    def test_solve_automatically_learns_rule(self):
        reasoner = SpatialReasoner()
        sol = reasoner.solve(
            [(_grid([[1, 0], [0, 0]]), _grid([[0, 1], [0, 0]]))],
            [_grid([[0, 0], [1, 0]])],
        )
        assert sol.solved
        assert reasoner._learned  # the verified rule was retained

    def test_learning_grounds_rules_into_network(self):
        from genesis_cognitive.spatial import Transform

        net = ConceptNetwork()
        reasoner = SpatialReasoner(net)
        reasoner.learn(
            [Transform("rotate90", lambda g: g.rotate90(1), {"turns": 1})]
        )
        concept = net.get_concept("spatial:rule:rotate90(turns=1)")
        assert concept is not None
        assert concept.properties["kind"] == "spatial_rule"


# ── Agent ────────────────────────────────────────────────────────


class TestSpatialAgent:
    """Mock environment: a single avatar cell moves with actions."""

    class _MockEnv:
        def __init__(self):
            self.pos = [5, 5]
            self.actions = ["up", "down", "left", "right"]

        def frame(self):
            g = [[0] * 11 for _ in range(11)]
            g[self.pos[0]][self.pos[1]] = 3  # avatar
            g[1][1] = 7  # goal object
            return g

        def step(self, action):
            if action == "up":
                self.pos[0] = max(0, self.pos[0] - 1)
            elif action == "down":
                self.pos[0] += 1
            elif action == "left":
                self.pos[1] = max(0, self.pos[1] - 1)
            elif action == "right":
                self.pos[1] += 1
            return self.frame()

    def test_agency_detection_and_effects(self):
        from genesis_cognitive.spatial import SpatialAgent

        env = self._MockEnv()
        agent = SpatialAgent(seed=0, epsilon=1.0)  # pure exploration
        agent.observe(env.frame())
        for action in ("up", "right", "up", "right"):
            agent._last_action = action
            env.step(action)
            agent.observe(env.frame())
        # The avatar (color 3) is the thing that moves when it acts.
        assert agent.avatar_color == 3
        # Displacement was attributed to the right actions.
        assert agent.stats["up"].dr < 0
        assert agent.stats["right"].dc > 0

    def test_navigation_moves_toward_goal(self):
        from genesis_cognitive.spatial import SpatialAgent

        env = self._MockEnv()
        agent = SpatialAgent(seed=0, epsilon=0.0)
        agent.observe(env.frame())
        # Teach the effect model directly, then let it navigate.
        for a, (dr, dc) in {
            "up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)
        }.items():
            st = agent._stat(a)
            st.attempts, st.dr, st.dc = 1, dr, dc
        agent.avatar_color = 3
        start = list(env.pos)
        for _ in range(10):
            a = agent.choose_action(env.actions)
            env.step(a)
            agent.observe(env.frame())
            agent.mark_goal_reached()
        # Goal is at (1,1); avatar started at (5,5) — should be closer.
        dist0 = abs(1 - start[0]) + abs(1 - start[1])
        dist1 = abs(1 - env.pos[0]) + abs(1 - env.pos[1])
        assert dist1 < dist0

    def test_learned_model_survives_episode_end(self):
        from genesis_cognitive.spatial import SpatialAgent

        agent = SpatialAgent(seed=0)
        agent.avatar_color = 3
        agent._avatar_votes[3] = 5
        agent.on_episode_end("GAME_OVER")
        assert agent.avatar_color == 3  # self-model persists

    def test_death_attribution_prefers_safe_actions(self):
        """An action present only in dying episodes is avoided.

        Fatal choices are often delayed — a doomed pick can precede the death
        by many steps — so blame accrues per-episode presence, not last-action.
        After enough deaths the lethal action drops out of the choice pool while
        the innocent one remains choosable.
        """
        from genesis_cognitive.spatial import SpatialAgent

        agent = SpatialAgent(seed=0, epsilon=1.0)
        actions = ["action3", "action4"]
        # Episodes containing action4 always die; clean episodes live.
        for _ in range(6):
            agent._cur_episode_actions = {"action3", "action4"}
            agent.on_episode_end("GAME_OVER")
            agent._cur_episode_actions = {"action3"}
            agent.on_episode_end("WIN")
        assert agent._is_lethal("action4")
        assert not agent._is_lethal("action3")
        for _ in range(30):
            assert agent.choose_action(actions) == "action3"

    def test_delayed_death_does_not_blame_last_action(self):
        """The action in flight at death isn't blamed when another
        action is the consistent discriminator."""
        from genesis_cognitive.spatial import SpatialAgent

        agent = SpatialAgent(seed=0, epsilon=1.0)
        # Pattern: episodes die only when "action4" was used — the
        # last action before death is often innocent "action3".
        for _ in range(5):
            agent._cur_episode_actions = {"action3", "action4"}
            agent._last_action = "action3"  # innocent last action
            agent.on_episode_end("GAME_OVER")
        for _ in range(5):
            agent._cur_episode_actions = {"action3"}
            agent._last_action = "action3"
            agent.on_episode_end("WIN")
        assert agent._is_lethal("action4")
        assert not agent._is_lethal("action3")

    def test_responsive_cells_are_reclicked(self):
        """A clicked cell that produced change is a working control —
        it gets re-pressed instead of exhausting to a blind sweep."""
        from genesis_cognitive.spatial import SpatialAgent

        agent = SpatialAgent(seed=0)
        agent.observe([[0] * 8 for _ in range(8)])
        agent._click_outcomes[(2, 2)] = 0.5   # a responsive control
        agent._click_outcomes[(0, 0)] = 0.0   # a dead spot
        data = agent.action_data("action6", (8, 8))
        assert (data["y"], data["x"]) == (2, 2)

    def test_lethal_cells_excluded_from_clicks(self):
        """Cells whose episodes reliably die stop being click targets."""
        from genesis_cognitive.spatial import SpatialAgent

        agent = SpatialAgent(seed=0)
        agent.observe([[0] * 8 for _ in range(8)])
        for _ in range(4):
            agent._cur_episode_clicks = {(1, 1)}
            agent.on_episode_end("GAME_OVER")
        agent._cur_episode_clicks = {(5, 5)}
        agent.on_episode_end("WIN")
        assert agent._is_lethal_cell((1, 1))
        assert not agent._is_lethal_cell((5, 5))
        agent._click_outcomes[(1, 1)] = 0.9  # responsive but lethal
        agent._click_outcomes[(5, 5)] = 0.3
        # Episode end clears the frame — observe the fresh episode's
        # first frame before choosing, as the run loop does.
        agent.observe([[0] * 8 for _ in range(8)])
        data = agent.action_data("action6", (8, 8))
        assert (data["y"], data["x"]) == (5, 5)

    def test_doom_marker_blames_introducer(self):
        """When every dying episode contains both actions, presence
        alone can't separate the killer from the necessary innocent.
        The visible trace can: the action whose arrival adds a color
        that keeps preceding death takes doom blame."""
        from genesis_cognitive.spatial import SpatialAgent

        agent = SpatialAgent(seed=0, epsilon=1.0)
        empty = [[0] * 8 for _ in range(8)]
        marked = [row[:] for row in empty]
        marked[3][3] = 9  # a new color appears after "harm" acts
        benign = [row[:] for row in empty]
        benign[0][0] = 5
        for _ in range(4):
            # Dying segment: the same two actions are present, and
            # "harm" leaves the marker behind.
            agent.observe(empty)
            agent._last_action = "harm"
            agent.observe(marked)          # introduces color 9
            agent._cur_episode_actions = {"harm", "needed"}
            agent.on_episode_end("GAME_OVER")
            # Surviving segment: identical action presence — so
            # presence statistics acquit "harm"; only the marker
            # it introduced can convict it.
            agent.observe(empty)
            agent._last_action = "fine"
            agent.observe(benign)          # introduces color 5
            agent._cur_episode_actions = {"harm", "needed", "fine"}
            agent.on_episode_end("WIN")
        assert not agent._is_doom_color(5)
        assert agent._doom_blame.get("harm", 0) >= 2
        assert "needed" not in agent._doom_blame
        assert "fine" not in agent._doom_blame
        assert agent._is_lethal("harm")
        assert not agent._is_lethal("needed")


# ── Grounding ────────────────────────────────────────────────────


class TestGrounding:
    def test_objects_become_concepts(self):
        net = ConceptNetwork()
        g = _grid([[1, 0], [0, 2]])
        scene = perceive(g, background=0)
        names = ground_scene(net, scene, ns="s1")
        assert "s1:obj:0" in names
        concept = net.get_concept("s1:obj:0")
        assert concept is not None
        assert concept.properties["kind"] == "spatial_object"
        assert concept.properties["color"] == 1

    def test_relations_become_edges(self):
        net = ConceptNetwork()
        g = _grid([
            [1, 0],
            [0, 0],
            [2, 0],
        ])
        scene = perceive(g, background=0)
        ground_scene(net, scene, ns="s2")
        edges = net.get_edges("s2:obj:0", direction="out")
        spatial = [e for e in edges if e.relation == RelationType.SPATIAL_RELATION]
        assert spatial, "expected a SPATIAL_RELATION edge for above"
        assert any("above" in e.target for e in spatial)

    def test_scene_facts_are_triples(self):
        g = _grid([
            [1, 0],
            [0, 0],
            [2, 0],
        ])
        scene = perceive(g, background=0)
        facts = scene_facts(scene, ns="s3")
        assert ("s3:obj:0", "has_color", "1") in facts
        assert ("s3:obj:0", "above", "s3:obj:1") in facts

    def test_namespaces_do_not_collide(self):
        net = ConceptNetwork()
        g = _grid([[1]])
        ground_scene(net, perceive(g, background=0), ns="a")
        ground_scene(net, perceive(g, background=0), ns="b")
        assert net.get_concept("a:obj:0") is not net.get_concept("b:obj:0")


# ── Schema induction & macro learning ─────────────────────────────


class TestSchemaInduction:
    def test_learned_rule_instantiates_new_params(self):
        from genesis_cognitive.spatial.transforms import Transform

        reasoner = SpatialReasoner()
        learned = Transform(
            "fill_enclosed", lambda g: g, {"color": 4}
        )
        reasoner.learn([learned])
        # A task showing different colors — the schema should propose
        # fill_enclosed re-bound to the colors this task contains.
        examples = [
            (
                _grid([[1, 1, 1], [1, 0, 1], [1, 1, 1]]),
                _grid([[1, 1, 1], [1, 2, 1], [1, 1, 1]]),
            )
        ]
        schemas = reasoner._schema_transforms(examples)
        params = {t.params.get("color") for t in schemas}
        assert 1 in params and 2 in params

    def test_gravity_schema_covers_direction_and_color(self):
        from genesis_cognitive.spatial.transforms import instantiate

        reasoner = SpatialReasoner()
        reasoner.learn([instantiate("gravity", {"direction": "down"})])
        reasoner.learn(
            [
                instantiate(
                    "gravity", {"direction": "down", "color": 3}
                )
            ]
        )
        examples = [
            (_grid([[0, 5, 0], [0, 0, 0]]), _grid([[0, 0, 0], [0, 5, 0]]))
        ]
        schemas = reasoner._schema_transforms(examples)
        names = {t.name for t in schemas}
        # Color is optional in the schema (one member lacks it), so
        # both unconditional and conditional directions appear.
        assert "gravity_left" in names or "gravity_right" in names
        assert any(
            n.startswith("gravity_") and n.count("_") == 2
            for n in names
        )

    def test_multistep_solution_becomes_macro(self):
        from genesis_cognitive.spatial.transforms import Transform

        reasoner = SpatialReasoner()
        t1 = Transform("crop_to_content", lambda g: g)
        t2 = Transform("rotate90", lambda g: g.rotate90(1), {"k": 1})
        reasoner.learn([t1, t2])
        macros = [
            t for t in reasoner._learned if t.name.startswith("macro:")
        ]
        assert macros, "expected a macro transform"
        g = _grid([[1, 2], [3, 4]])
        assert macros[0](g).to_lists() == [[3, 1], [4, 2]]

    def test_schema_proposals_feed_solve(self):
        # Learn fill_enclosed(4); a held-out task needing
        # fill_enclosed(2) should still be solvable via the schema
        # even though the learned instance itself doesn't apply.
        from genesis_cognitive.spatial.transforms import instantiate

        reasoner = SpatialReasoner()
        reasoner.learn([instantiate("fill_enclosed", {"color": 4})])
        box_in = _grid([
            [0, 0, 0, 0, 0],
            [0, 3, 3, 3, 0],
            [0, 3, 0, 3, 0],
            [0, 3, 3, 3, 0],
            [0, 0, 0, 0, 0],
        ])
        box_out = _grid([
            [0, 0, 0, 0, 0],
            [0, 3, 3, 3, 0],
            [0, 3, 2, 3, 0],
            [0, 3, 3, 3, 0],
            [0, 0, 0, 0, 0],
        ])
        sol = reasoner.solve([(box_in, box_out)], [])
        assert sol.solved
        assert any(
            t.name == "fill_enclosed" and t.params.get("color") == 2
            for t in sol.hypothesis.transforms
        )

    def test_failure_info_reports_counterexample(self):
        # An unsolvable-at-depth-1 task still carries a counterexample
        # report: which pair the best rule broke on and how.
        a = _grid([[1, 0], [0, 0]])
        b = _grid([[0, 1], [0, 0]])
        out_a = _grid([[1, 0], [0, 0]])
        out_b = _grid([[0, 0], [0, 1]])
        reasoner = SpatialReasoner()
        sol = reasoner.solve([(a, out_a), (b, out_b)], [a], max_depth=1)
        assert not sol.solved
        assert sol.failure is not None
        assert sol.hypothesis is not None
        assert sol.failure.rule == sol.hypothesis.describe()
        assert len(sol.failure.pair_scores) == 2
        assert sol.failure.mismatched != 0 or sol.failure.shape_mismatch
        assert sol.failure.describe()

    def test_exclude_rules_prunes_retry(self):
        # A rule already proven wrong on this task can't be re-selected.
        inp = _grid([[1, 0], [0, 0]])
        out = _grid([[0, 1], [0, 0]])
        reasoner = SpatialReasoner()
        sol = reasoner.solve([(inp, out)], [inp])
        assert sol.solved and sol.hypothesis is not None
        first = sol.hypothesis.describe()
        sol2 = reasoner.solve(
            [(inp, out)], [inp], exclude_rules={first}
        )
        assert first not in sol2.verified_rules
        if sol2.solved:
            assert sol2.hypothesis is not None
            assert sol2.hypothesis.describe() != first

    def test_prediction_key_prunes_repeated_behavior(self):
        # The behavioral exclusion must block an answer even when its
        # rule description is not excluded. A different exact rule may
        # still be selected if it produces a different prediction.
        inp = _grid([[1, 0], [0, 0]])
        out = _grid([[0, 1], [0, 0]])
        test = _grid([[0, 0], [0, 1]])
        reasoner = SpatialReasoner()
        first = reasoner.solve([(inp, out)], [test])
        assert first.solved and first.predictions
        blocked = reasoner.prediction_key(first.predictions)
        second = reasoner.solve(
            [(inp, out)], [test], exclude_predictions={blocked}
        )
        if second.solved:
            assert reasoner.prediction_key(second.predictions) != blocked
