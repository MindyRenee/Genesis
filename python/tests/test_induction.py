"""Tests for frontal_lobe.reasoning.induction — exemplar→rule abstraction.

The benchmark case: paired glyph panels forming a dictionary, a query
strip of rotated glyphs, translate by substitution.
"""

from __future__ import annotations

from frontal_lobe.reasoning.induction import (
    AbstractLevel,
    ExampleInductor,
    RuleStatus,
)
from parietal_lobe.exemplars import ExemplarSession, _Panel, _PanelWriter
from parietal_lobe.grid import Grid
from parietal_lobe.strips import orbit, segment


def _g(rows: list[list[int]]) -> Grid:
    return Grid.from_lists(rows)


def _tile(seed: int, fg: int = 5, bg: int = 0, n: int = 5) -> Grid:
    """A deterministic glyph, unique under all dihedral transforms:
    a fixed top-left notch (orientation landmark — a transformed
    glyph's notch lands elsewhere, so it can never equal another
    seed's canonical pattern) plus a seed-positioned mark."""
    rows = [[bg] * n for _ in range(n)]
    rows[0][0] = fg
    rows[0][1] = fg  # notch — breaks all rotation/reflection symmetry
    mark_row = 2 if fg >= 8 else 3  # consequent family: different shape
    rows[mark_row][seed % n] = fg  # seed mark — distinguishes seeds
    return _g(rows)


def _strip(blocks: list[Grid], wall: int) -> Grid:
    h = blocks[0].height
    rows: list[list[int]] = [
        [wall] * (sum(b.width for b in blocks) + len(blocks) + 1) for _ in range(h)
    ]
    c = 1
    for b in blocks:
        for r in range(h):
            rows[r][c : c + b.width] = list(b.rows()[r])
        c += b.width + 1
    return _g(rows)


class TestFeatureRules:
    def test_rotation_induced_and_applied(self) -> None:
        ind = ExampleInductor()
        a1, a2 = _tile(1), _tile(2)
        ind.observe(a1, a1.rotate90(1))
        ind.observe(a2, a2.rotate90(1))
        rule = ind.induce()
        assert rule is not None and rule.describe() == "rot(1)"
        novel = _tile(3)
        assert ind.apply(novel) == novel.rotate90(1)
        assert ind.status == RuleStatus.VERIFIED

    def test_recolor_map_learned_from_pairs(self) -> None:
        ind = ExampleInductor()
        a = _g([[1, 1, 2], [2, 0, 1], [0, 2, 2]])
        b = _g([[4, 4, 7], [7, 0, 4], [0, 7, 7]])
        ind.observe(a, b)
        a2 = _g([[2, 2, 1], [1, 0, 2], [0, 1, 1]])
        b2 = _g([[7, 7, 4], [4, 0, 7], [0, 4, 4]])
        ind.observe(a2, b2)
        rule = ind.induce()
        assert rule is not None and rule.kind == "recolor"
        q = _g([[1, 2, 0], [2, 1, 0], [0, 0, 1]])
        assert ind.apply(q) == _g([[4, 7, 0], [7, 4, 0], [0, 0, 4]])

    def test_identity_when_pairs_are_equal(self) -> None:
        ind = ExampleInductor()
        t = _tile(4)
        ind.observe(t, t)
        assert ind.induce() is not None
        assert ind.apply(_tile(5)) == _tile(5)

    def test_concrete_rule_wins_over_substitution(self) -> None:
        # Rostrocaudal ordering: when a feature-level rule covers the
        # pairs, the relational candidate must not win.
        ind = ExampleInductor()
        a1, a2 = _tile(1, fg=5), _tile(2, fg=5)
        ind.observe(a1, a1.rotate90(2))
        ind.observe(a2, a2.rotate90(2))
        rule = ind.induce()
        assert rule is not None
        assert rule.level == AbstractLevel.FEATURE


class TestSubstitution:
    def _dictionary(self, ind: ExampleInductor, n: int = 4) -> list[Grid]:
        """Build an n-glyph exemplar dictionary; returns antecedents."""
        antes = [_tile(10 + i, fg=5, bg=0) for i in range(n)]
        outs = [_tile(30 + i, fg=8, bg=0) for i in range(n)]
        for a, o in zip(antes, outs, strict=True):
            ind.observe(a, o)
        return antes

    def test_translate_rotated_query(self) -> None:
        """Query glyphs arrive rotated; matching must tolerate tilt."""
        ind = ExampleInductor()
        antes = self._dictionary(ind)
        rule = ind.induce()
        assert rule is not None and rule.kind == "substitution"

        # Query: strip of 3 glyphs — one exact, two rotated.
        query = _strip(
            [antes[1], antes[3].rotate90(1), antes[0].rotate90(3)],
            wall=7,
        )
        out = ind.apply(query)
        assert out is not None

        # Decode: split the output on wall columns, match each to a
        # consequent — the emitted phrase must be B1 B3 B0.
        blocks = segment(out, separator=7)
        assert blocks is not None and len(blocks) == 3
        expected = [
            ind.store.pairs[1].consequent,
            ind.store.pairs[3].consequent,
            ind.store.pairs[0].consequent,
        ]
        for i, (blk, exp) in enumerate(zip(blocks, expected, strict=True)):
            assert blk == exp, f"block {i}: got wrong consequent"

    def test_substitution_first_rule_wins(self) -> None:
        """Duplicate antecedents: first dictionary entry applies."""
        ind = ExampleInductor()
        a = _tile(10, fg=5)
        o1, o2 = _tile(30, fg=8), _tile(31, fg=8)
        ind.observe(a, o1)
        ind.observe(a, o2)
        ind.observe(_tile(11, fg=5), _tile(32, fg=8))
        ind.induce()
        assert ind.apply(a) == o1

    def test_exemplar_fallback_when_no_rule(self) -> None:
        """Query matches a single antecedent → its consequent back."""
        ind = ExampleInductor()
        antes = self._dictionary(ind, n=3)
        ind.induce()
        # Single-glyph query, rotated — exemplar route must answer.
        assert ind.apply(antes[2].rotate90(2)) == ind.store.pairs[2].consequent

    def test_unmatched_block_returns_silence(self) -> None:
        ind = ExampleInductor()
        antes = self._dictionary(ind, n=3)
        ind.induce()
        foreign = _tile(99, fg=9, bg=1)
        query = _strip([antes[0], foreign], wall=7)
        assert ind.apply(query) is None


class TestRuleChange:
    def test_contradiction_flags_research(self) -> None:
        ind = ExampleInductor()
        a = _tile(1)
        ind.observe(a, a.rotate90(1))
        ind.observe(_tile(2), _tile(2).rotate90(1))
        rule = ind.induce()
        assert rule is not None and not ind.needs_search
        # The world changes its mind: same antecedent, new consequent.
        ind.observe(a, a.rotate90(3))
        assert ind.needs_search
        assert ind.violations >= 1

    def test_reinduction_picks_covering_rule(self) -> None:
        ind = ExampleInductor()
        a, b = _tile(1), _tile(2)
        ind.observe(a, a.rotate90(1))
        ind.observe(b, b.rotate90(1))
        ind.induce()
        # Overwhelming counter-evidence → a different rule must win.
        c, d = _tile(3), _tile(4)
        ind.observe(c, c.rotate90(2))
        ind.observe(d, d.rotate90(2))
        # Old rule no longer covers all pairs.
        assert ind.rule is not None
        assert not all(ind.rule.predict(p.antecedent) == p.consequent for p in ind.store.pairs)


class TestSegmentation:
    def test_strip_roundtrip(self) -> None:
        blocks = [_tile(i, fg=5) for i in range(3)]
        strip = _strip(blocks, wall=7)
        parts = segment(strip, separator=7, exemplar_shape=(5, 5))
        assert parts == blocks

    def test_no_walls_returns_none(self) -> None:
        assert segment(_tile(1, fg=5), separator=7) is None


class TestPersistence:
    def test_roundtrip(self) -> None:
        ind = ExampleInductor()
        ind.observe(_tile(1), _tile(1).rotate90(1))
        ind.observe(_tile(2), _tile(2).rotate90(1))
        ind.induce()
        data = ind.to_dict()
        ind2 = ExampleInductor.from_dict(data)
        assert len(ind2.store.pairs) == 2
        assert ind2.apply(_tile(3)) == _tile(3).rotate90(1)


# ── ExemplarSession: parietal scene-parse → frontal induction ──


def _frame(top: int, left: int, border: int, interior: Grid, scene: list[list[int]]) -> None:
    """Paint a bordered panel into a scene grid."""
    h, w = interior.height + 2, interior.width + 2
    for r in range(h):
        for c in range(w):
            scene[top + r][left + c] = border
    for r in range(interior.height):
        for c in range(interior.width):
            scene[top + 1 + r][left + 1 + c] = interior.at(r, c)


def _demo_scene() -> tuple[list[list[int]], list[Grid], list[Grid]]:
    """A demonstration scene: 3 dict pairs up top, query strip below."""
    bg = 1
    scene = [[bg] * 64 for _ in range(64)]
    antes = [_tile(10 + i, fg=5) for i in range(3)]
    cons = [_tile(30 + i, fg=8) for i in range(3)]
    x = 4
    for a, o in zip(antes, cons, strict=True):
        _frame(4, x, 10, a, scene)
        _frame(4, x + 8, 7, o, scene)
        x += 20
    query = _strip([antes[1], antes[0].rotate90(1), antes[2]], wall=10)
    _frame(20, 4, 10, query, scene)
    _frame(30, 4, 7, _strip([_tile(61, fg=7)] * 3, wall=7), scene)
    return scene, antes, cons


class TestExemplarSession:
    def test_scene_pairs_feed_inductor(self) -> None:
        scene, _antes, _cons = _demo_scene()
        sess = ExemplarSession()
        g = _g(scene)
        sess.observe(g)
        # A demonstration must persist — one frame only accrues votes.
        assert len(sess.inductor.store.pairs) == 0
        sess.observe(g)
        assert len(sess.inductor.store.pairs) == 3
        # The induced rule covers all three pairs.
        assert sess.inductor.induce() is not None

    def test_expected_picks_antecedent_framed_query(self) -> None:
        scene, _antes, cons = _demo_scene()
        sess = ExemplarSession()
        g = _g(scene)
        sess.observe(g)
        sess.observe(g)
        target = sess.expected()
        assert target is not None
        # Query frame is the antecedent color (10), not the answer's (7).
        assert target.panel.border == 10
        # Prediction decodes to B1 B0 B2 under the same wall layout.
        blocks = segment(target.predicted, separator=10)
        assert blocks is not None and len(blocks) == 3
        assert blocks[0] == cons[1]
        assert blocks[1] == cons[0]
        assert blocks[2] == cons[2]

    def test_empty_scene_is_silent(self) -> None:
        sess = ExemplarSession()
        sess.observe(_g([[1] * 32 for _ in range(32)]))
        assert sess.expected() is None
        assert len(sess.inductor.store.pairs) == 0

    def test_transient_pair_never_commits(self) -> None:
        """A pair visible on one frame only is noise — animation or
        a transition artifact — and must never reach the store or
        the frame-color schema."""
        scene, _antes, _cons = _demo_scene()
        sess = ExemplarSession()
        g = _g(scene)
        sess.observe(g)
        sess.observe(g)
        assert len(sess.inductor.store.pairs) == 3
        # One noisy frame: a junk pair appears beside the query row.
        noisy = [row[:] for row in scene]
        _frame(20, 50, 2, _tile(71, fg=8, n=3), noisy)
        _frame(20, 56, 2, _tile(72, fg=8, n=3), noisy)
        sess.observe(_g(noisy))
        # Back to the clean scene — the junk never returns.
        sess.observe(g)
        sess.observe(g)
        assert len(sess.inductor.store.pairs) == 3
        assert 2 not in sess._ante_borders and 2 not in sess._cons_borders

    def test_export_import_restores_session_learning(self) -> None:
        """Second episode: borders, committed pairs, and verb skills
        ride the export; live scene state re-derives from frames."""
        scene, _antes, cons = _demo_scene()
        sess = ExemplarSession()
        g = _g(scene)
        sess.observe(g)
        sess.observe(g)
        import json

        data = json.loads(json.dumps(sess.export_learning()))
        restored = ExemplarSession()
        restored.import_learning(data)
        assert restored._ante_borders == sess._ante_borders
        assert restored._cons_borders == sess._cons_borders
        assert restored._seen_pairs == sess._seen_pairs
        assert len(restored.inductor.store.pairs) == 3
        # The new episode's first frame re-derives the panels and
        # the imported evidence predicts — no re-learning frames.
        restored.observe(g)
        target = restored.expected()
        assert target is not None and target.panel.border == 10
        blocks = segment(target.predicted, separator=10)
        assert blocks is not None
        assert [blocks[0], blocks[1], blocks[2]] == [cons[1], cons[0], cons[2]]

    def test_pair_votes_survive_export(self) -> None:
        """A pair seen once before shutdown commits on the next
        session's first sighting — votes are evidence too."""
        scene, _antes, _cons = _demo_scene()
        sess = ExemplarSession()
        g = _g(scene)
        sess.observe(g)
        assert len(sess.inductor.store.pairs) == 0 and sess._pair_votes
        import json

        restored = ExemplarSession()
        restored.import_learning(json.loads(json.dumps(sess.export_learning())))
        restored.observe(g)
        assert len(restored.inductor.store.pairs) == 3

    def test_heterogeneous_dictionary_still_induces(self) -> None:
        """A mixed-size store is still a lookup table — matching is
        per-block, so uniform shapes are not required."""
        ind = ExampleInductor()
        a1, a2 = _tile(1), _tile(2)
        ind.observe(a1, _tile(11, fg=8))
        ind.observe(a2, _tile(12, fg=8))
        ind.observe(_tile(3, n=4), _tile(13, fg=8, n=4))
        rule = ind.induce()
        assert rule is not None and rule.kind == "substitution"
        assert ind.apply(a2) == _tile(12, fg=8)


# ── _PanelWriter — the hand ────────────────────────────────────
#
# The writer's world is a mutable answer panel the test mutates
# between observes: the test plays the environment, the writer
# learns which of its acts does what.


def _glyph(k: int, n: int = 3) -> Grid:
    """A small dihedral-unique glyph: 5-marks on 7. A fixed
    landmark row plus one mark in *every* column — an empty
    column would read as a wall and poison the strip parse."""
    rows = [[7] * n for _ in range(n)]
    rows[0][0] = 5
    rows[0][1] = 5  # landmark — anchors orientation
    rows[1 + (k % 2)][0] = 5
    rows[1 + ((k // 2) % 2)][1] = 5
    rows[1 + ((k // 4) % 2)][2] = 5
    return _g(rows)


class _WriterWorld:
    """An answer strip the writer can act on.

    Three 3-wide slots in a 7-walled strip inside a bordered panel;
    a 0-marker sits above the selected slot. 'w' cycles the selected
    slot's variant; 'left'/'right' move the marker; 'x' does nothing
    visible to the panel.
    """

    TOP, LEFT = 5, 2

    def __init__(self) -> None:
        self.variants = [_glyph(k) for k in range(6)]
        self.idx = [0, 3, 5]
        self.slots = [self.variants[i] for i in self.idx]
        self.cursor = 0
        self.scene = [[3] * 20 for _ in range(20)]
        self._repaint()

    def _repaint(self) -> None:
        _frame(self.TOP, self.LEFT, 7, _strip(self.slots, wall=7), self.scene)
        for r in (3, 4):
            for c in range(self.LEFT - 1, self.LEFT + 14):
                self.scene[r][c] = 3
        c0 = self.LEFT + 1 + self.cursor * 4
        for c in range(c0, c0 + 3):
            self.scene[3][c] = 0
        self.scene[4][c0] = 0

    def grid(self) -> Grid:
        return _g(self.scene)

    def panel(self) -> _Panel:
        return _Panel(
            top=self.TOP,
            left=self.LEFT,
            height=5,
            width=13,
            border=7,
            interior=_strip(self.slots, wall=7),
        )

    def press(self, writer: _PanelWriter, action: str, *, effect: bool = True) -> None:
        writer.note_action(action)
        if effect:
            if action == "right":
                self.cursor = min(2, self.cursor + 1)
            elif action == "left":
                self.cursor = max(0, self.cursor - 1)
            elif action == "w":
                self.idx[self.cursor] = (self.idx[self.cursor] + 1) % len(self.variants)
                self.slots[self.cursor] = self.variants[self.idx[self.cursor]]
        self._repaint()


def _learned_writer() -> tuple[_WriterWorld, _PanelWriter]:
    world = _WriterWorld()
    w = _PanelWriter()
    w.observe(world.grid(), world.panel())
    world.press(w, "right")
    w.observe(world.grid(), world.panel())
    world.press(w, "left")
    w.observe(world.grid(), world.panel())
    world.press(w, "w")
    w.observe(world.grid(), world.panel())
    return world, w


class TestPanelWriter:
    def test_marker_reads_selected_slot(self) -> None:
        world = _WriterWorld()
        w = _PanelWriter()
        w.observe(world.grid(), world.panel())
        assert w._bracket == 0
        assert w._slots is not None and len(w._slots) == 3

    def test_verbs_learned_from_effects(self) -> None:
        _world, w = _learned_writer()
        assert w._moves == {"right": 1, "left": -1}
        assert w._writes == ["w"]

    def test_act_moves_to_mismatch_then_writes(self) -> None:
        world, w = _learned_writer()
        targets = list(world.slots)
        targets[1] = world.variants[4]
        avail = ["w", "right", "left", "x"]
        assert w.act(targets, avail) == "right"
        world.press(w, "right")
        w.observe(world.grid(), world.panel())
        assert w.act(targets, avail) == "w"

    def test_no_target_is_silence(self) -> None:
        world, w = _learned_writer()
        assert w.act(list(world.slots), ["w", "right", "left"]) is None

    def test_probes_the_unseen_verb(self) -> None:
        world = _WriterWorld()
        w = _PanelWriter()
        w.observe(world.grid(), world.panel())
        targets = list(world.slots)
        targets[0] = world.variants[1]
        first = w.act(targets, ["w", "right", "left"])
        assert first == "w"  # first unprobed action

    def test_abstains_when_all_verbs_exhausted(self) -> None:
        world, w = _learned_writer()
        w._inert.update({"x"})
        targets = list(world.slots)
        targets[1] = world.variants[4]
        # 'w' is a write verb at bracket 0 but target mismatch is slot 1;
        # bracket moves are known → it still acts. Force abstention:
        w._moves.clear()
        w._inert.update({"right", "left"})
        assert w.act(targets, ["w", "right", "left", "x"]) is None

    def test_violated_prediction_withdraws_and_counts(self) -> None:
        world, w = _learned_writer()
        targets = list(world.slots)
        targets[0] = world.variants[4]
        choice = w.act(targets, ["w", "right", "left"])
        assert choice == "w"
        world.press(w, "w", effect=False)  # the verb failed her
        w.observe(world.grid(), world.panel())
        assert w.errors == 1
        assert w.evidence["w"] == [0, 1]
        assert "w" not in w._writes

    def test_confirmed_choice_supports_verb(self) -> None:
        world, w = _learned_writer()
        targets = list(world.slots)
        targets[0] = world.variants[4]
        assert w.act(targets, ["w", "right", "left"]) == "w"
        world.press(w, "w")
        w.observe(world.grid(), world.panel())
        assert w.errors == 0
        assert w.evidence["w"] == [1, 0]

    def test_unrelated_action_is_inert(self) -> None:
        world, w = _learned_writer()
        world.press(w, "x")
        w.observe(world.grid(), world.panel())
        assert "x" in w._inert


class TestVisualIdentityProbe:
    """The ventral-stream assist: learned shape concepts may bridge
    tiles exact matching misses — without ever overriding evidence."""

    def _pair_rule(self, probe=None):
        from frontal_lobe.reasoning.induction import _Pair, _Substitution

        a, c = _tile(10, fg=5), _tile(30, fg=8)
        return _Substitution([_Pair(a, c)], match_dihedral=True, glyph_probe=probe), a

    def test_exact_match_needs_no_probe(self) -> None:
        rule, ante = self._pair_rule()
        assert rule._match_tile(ante, ante)

    def test_same_concept_bridges_a_miss(self) -> None:
        def probe(_tile):
            return "glyph:x"

        rule, ante = self._pair_rule(probe)
        other = _tile(99, fg=2)  # no exact/dihedral relation to ante
        assert not any(v == other for v in orbit(ante))
        assert rule._match_tile(ante, other)

    def test_different_concepts_do_not_match(self) -> None:
        names = iter(["glyph:a", "glyph:b"])
        rule, ante = self._pair_rule(lambda _t: next(names))
        assert not rule._match_tile(ante, _tile(99, fg=2))

    def test_unrecognized_tiles_do_not_match(self) -> None:
        rule, ante = self._pair_rule(lambda _t: None)
        assert not rule._match_tile(ante, _tile(99, fg=2))

    def test_no_probe_is_exact_only(self) -> None:
        rule, ante = self._pair_rule()
        assert not rule._match_tile(ante, _tile(99, fg=2))

    def test_set_glyph_probe_stamps_standing_rule(self) -> None:
        ind = ExampleInductor()
        for a, o in zip([_tile(1), _tile(2)], [_tile(3), _tile(4)], strict=True):
            ind.observe(a, o)
        ind.observe(_tile(5), _tile(6))
        rule = ind.induce()
        assert rule is not None and rule.kind == "substitution"
        ind.set_glyph_probe(lambda _t: "glyph:z")
        assert rule.glyph_probe is not None  # the standing rule isn't blind
        ind.set_glyph_probe(None)
        assert rule.glyph_probe is None


class TestCompetenceBridge:
    """The session speaks the shared TaskCompetence protocol itself:
    recognize the format, report classified-verb transitions,
    consolidate a verified episode into a grounded skill."""

    def _scene_session(self, competence=None):
        scene, _antes, _cons = _demo_scene()
        sess = ExemplarSession()
        sess.competence = competence
        g = _g(scene)
        sess.observe(g)
        sess.observe(g)
        return sess, scene, g

    def test_engaged_only_with_committed_pairs(self) -> None:
        sess = ExemplarSession()
        assert not sess.engaged()
        sess2, _s, _g = self._scene_session()
        assert sess2.engaged()

    def test_state_and_schema_vocabulary(self) -> None:
        sess, _s, _g = self._scene_session()
        state = sess.state()
        assert state["pair.count"] == 3
        assert state["rule.kind"] in ("none", "substitution")
        schema = sess.schema_state()
        # Schema is format identity — no phase variables inside.
        assert "rule.kind" not in schema
        assert schema["tile.w"] == 5

    def test_verb_classification_names_semantic_verbs(self) -> None:
        sess, _s, _g = self._scene_session()
        w = sess.writer
        w._writes.append("1")
        w._moves["4"] = 1
        w._inert.add("3")
        assert sess.verb_of("1") == "write"
        assert sess.verb_of("4") == "move"
        assert sess.verb_of("3") == "inert"
        assert sess.verb_of("2") is None

    def test_recognize_and_transition_ledger(self) -> None:
        from frontal_lobe.reasoning.competence import TaskCompetence

        comp = TaskCompetence()
        sess, _s, g = self._scene_session(comp)
        # Recognition happened once the format engaged.
        assert sess._ctx is not None
        assert comp.schema_count == 1
        schema = next(iter(comp.schemas.values()))
        assert schema.signature.domain == "world.exemplar"
        # A press classified as write becomes a verb transition.
        sess.note_action("1")
        sess.writer._writes.append("1")
        sess.observe(g)
        verbs = set(schema.operators)
        assert "write" in verbs or "probe" in verbs
        assert comp.transition_count == 1

    def test_episode_consolidates_and_grounds(self) -> None:
        from frontal_lobe.reasoning.competence import TaskCompetence
        from temporal_lobe.concepts import ConceptNetwork

        net = ConceptNetwork()
        comp = TaskCompetence(network=net)
        sess, _s, g = self._scene_session(comp)
        sess.note_action("1")
        sess.writer._writes.append("1")
        sess.observe(g)
        sess.inductor.induce()
        # The tick notes a rule reaching VERIFIED on the frame after
        # it is induced — the same order live play produces.
        sess.observe(g)
        sess.end_episode(success=True)
        # One skill, externally verified, grounded into the network.
        assert len(comp.skills) == 1
        skill = next(iter(comp.skills.values()))
        assert skill.verification == "external"
        labels = skill.step_labels()
        assert "read_demonstrations" in labels[0]
        assert any("induce_rule" in label for label in labels)
        concept = net.get_concept(f"skill:{skill.skill_id}")
        assert concept is not None
        assert concept.properties["verification"] == "external"
        # Milestones: the verified rule + the answer that held.
        notes = [t for t, _s in sess.drain_milestones()]
        assert any("induced" in t for t in notes)
        assert any("held" in t for t in notes)
        assert sess.drain_milestones() == []  # drained

    def test_failed_episode_no_skill_but_ledger(self) -> None:
        from frontal_lobe.reasoning.competence import TaskCompetence

        comp = TaskCompetence()
        sess, _s, g = self._scene_session(comp)
        sess.note_action("1")
        sess.writer._writes.append("1")
        sess.observe(g)
        sess.end_episode(success=False)
        assert len(comp.skills) == 0
        schema = next(iter(comp.schemas.values()))
        assert schema.failures == 1
        assert any("unanswered" in t for t, _s in sess.drain_milestones())

    def test_no_competence_is_silent(self) -> None:
        sess, _s, g = self._scene_session(None)
        sess.note_action("1")
        sess.observe(g)
        sess.end_episode(success=True)  # must not raise
        assert sess.drain_milestones() != []

    def test_glyph_ids_persist_round_trip(self) -> None:
        sess, _s, _g = self._scene_session()
        tile = _tile(10, fg=5)
        sess.glyph_ids[tile.rows()] = "glyph:abc123"
        import json

        data = json.loads(json.dumps(sess.export_learning()))
        restored = ExemplarSession()
        restored.import_learning(data)
        assert restored.glyph_ids[tile.rows()] == "glyph:abc123"

    def test_glyph_tiles_are_committed_evidence_only(self) -> None:
        sess, _s, _g = self._scene_session()
        tiles = sess.glyph_tiles()
        # 3 pairs × (ante + cons) = 6 unique tiles, all 5×5.
        assert len(tiles) == 6
        assert all(t.shape == (5, 5) for t in tiles)

    def test_glyph_pairs_are_one_to_one(self) -> None:
        sess, _s, _g = self._scene_session()
        pairs = sess.glyph_pairs()
        assert len(pairs) == 3
        assert all(a.shape == c.shape == (5, 5) for a, c in pairs)


class TestConceptGrounding:
    """The format's vocabulary lives in her concept network —
    adopted where the word exists, minted where it doesn't, and
    every relation lands only where it was observed."""

    def _net_session(self):
        from temporal_lobe.concepts import ConceptNetwork

        net = ConceptNetwork()
        scene, _antes, _cons = _demo_scene()
        sess = ExemplarSession()
        sess.network = net
        g = _g(scene)
        sess.observe(g)
        sess.observe(g)
        return sess, net, g

    def test_vocabulary_is_claimed_on_engagement(self) -> None:
        _sess, net, _g = self._net_session()
        for word in (
            "pair",
            "query",
            "answer",
            "tile",
            "slot",
            "write",
            "move",
            "probe",
            "rule",
            "verdict",
        ):
            assert net.get_concept(word) is not None, word
            assert net.get_concept(word).properties.get("domain") == "world.exemplar"

    def test_existing_word_is_adopted_not_duplicated(self) -> None:
        from temporal_lobe.concepts import ConceptNetwork

        net = ConceptNetwork()
        prior = net.add_concept("pair", origin="conversation")
        scene, _a, _c = _demo_scene()
        sess = ExemplarSession()
        sess.network = net
        sess.observe(_g(scene))
        sess.observe(_g(scene))
        # Same node — the word she knew gained the operational role.
        assert net.get_concept("pair").id == prior.id
        assert "pair" in sess._claimed

    def test_no_claims_before_evidence(self) -> None:
        from temporal_lobe.concepts import ConceptNetwork

        net = ConceptNetwork()
        sess = ExemplarSession()
        sess.network = net
        sess.observe(_g([[1] * 32 for _ in range(32)]))  # empty scene
        assert net.get_concept("pair") is None
        assert sess._claimed == set()

    def test_edges_land_at_evidence_sites(self) -> None:
        from temporal_lobe.concepts import RelationType

        _sess, net, _g = self._net_session()
        rels = {(e.source, e.target, e.relation) for e in net.get_edges("pair")}
        assert ("pair", "demonstration", RelationType.IS_A) in rels
        assert ("pair", "tile", RelationType.PART_OF) in rels
        # No relation asserted before its evidence: no rule yet.
        assert net.get_edges("rule") == []

    def test_verb_classification_grounds_the_verb(self) -> None:
        from temporal_lobe.concepts import RelationType

        sess, net, g = self._net_session()
        sess.writer._writes.append("1")
        sess.observe(g)  # diff noticed -> write~slot edge lands
        rels = {(e.source, e.target, e.relation) for e in net.get_edges("write")}
        assert ("write", "slot", RelationType.RELATED_TO) in rels

    def test_state_keys_are_concept_names(self) -> None:
        sess, net, _g = self._net_session()
        for key in sess.state():
            base = key.split(".")[0]
            assert net.get_concept(base) is not None, key

    def test_standalone_session_keeps_plain_words(self) -> None:
        scene, _a, _c = _demo_scene()
        sess = ExemplarSession()  # no network — words pass through
        g = _g(scene)
        sess.observe(g)
        sess.observe(g)
        assert sess.state()["pair.count"] == 3
        sess.writer._writes.append("1")
        assert sess.verb_of("1") == "write"
