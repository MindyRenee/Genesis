"""Classification world + competence adapter — the rule family.

"Does this fit the rule?" / "What's the rule?" — the invented-word
games ("a tessel is a machine with exactly two of..."), odd-one-out,
membership sorting. Two honest modes share one world: ``apply`` reads
a stated predicate and labels items against it; ``induce`` has no
stated rule and must learn which features predict membership from
worked examples plus per-guess oracle feedback.
"""

from __future__ import annotations

import json

from genesis_cognitive.classification import (
    Item,
    RuleAgent,
    RuleGame,
    eval_predicate,
    valid_predicate,
)
from genesis_cognitive.reasoning import TaskCompetence
from genesis_cognitive.spatial.practice import (
    SpatialPractice,
    normalize_offered,
)


def _tessel_predicate() -> dict:
    return {"op": "exactly", "of": ["lens", "spring", "dial"], "count": 2}


def _tessel_items() -> list[Item]:
    return [
        Item(0, "alpha", ("lens", "spring")),
        Item(1, "beta", ("lens", "spring", "dial")),
        Item(2, "gamma", ("dial",)),
        Item(3, "delta", ("spring", "dial")),
    ]


class TestPredicates:
    def test_eval_all_ops(self) -> None:
        attrs = frozenset({"lens", "spring"})
        of = ["lens", "spring", "dial"]
        assert eval_predicate({"op": "exactly", "of": of, "count": 2}, attrs)
        assert not eval_predicate(
            {"op": "exactly", "of": of, "count": 3}, attrs
        )
        assert eval_predicate({"op": "at_least", "of": of, "count": 2}, attrs)
        assert eval_predicate({"op": "at_most", "of": of, "count": 3}, attrs)
        assert eval_predicate({"op": "any_of", "of": of}, attrs)
        assert not eval_predicate({"op": "all_of", "of": of}, attrs)
        assert eval_predicate({"op": "all_of", "of": ["lens", "spring"]}, attrs)
        assert not eval_predicate({"op": "none_of", "of": of}, attrs)
        assert eval_predicate({"op": "none_of", "of": ["crank"]}, attrs)

    def test_valid_predicate_rejects_garbage(self) -> None:
        assert not valid_predicate(None)
        assert not valid_predicate({"op": "bogus", "of": ["x"]})
        assert not valid_predicate({"op": "all_of", "of": []})
        assert not valid_predicate(
            {"op": "exactly", "of": ["a"], "count": 5}
        )
        assert valid_predicate({"op": "any_of", "of": ["x"]})


class TestRuleGame:
    def test_apply_mode_oracle(self) -> None:
        world = RuleGame(_tessel_predicate(), _tessel_items())
        assert world.mismatches() == 4
        out = world.guess("alpha", True)
        assert out == {"correct": True, "truth": True, "item": "alpha"}
        assert world.mismatches() == 3
        # Wrong guess: honest feedback reveals the truth, costs the move.
        out = world.guess("beta", True)
        assert out == {"correct": False, "truth": False, "item": "beta"}
        assert world.mismatches() == 3  # still pending — retake allowed
        out = world.guess("beta", False)
        assert (out or {})["correct"]
        assert world.complete() is False

    def test_induce_mode_answer_key(self) -> None:
        items = [
            Item(0, "a", ("x", "y"), True),
            Item(1, "b", ("x",), False),
        ]
        world = RuleGame(None, items)
        assert (world.guess("a", True) or {})["correct"]
        assert not (world.guess("b", True) or {})["correct"]
        assert (world.guess("b", False) or {})["correct"]
        assert world.complete()

    def test_unnamed_item_returns_none(self) -> None:
        world = RuleGame(_tessel_predicate(), _tessel_items())
        assert world.guess("nobody", True) is None


class TestRuleAgent:
    def test_apply_solves_by_reading_the_rule(self) -> None:
        world = RuleGame(_tessel_predicate(), _tessel_items())
        res = RuleAgent(seed=1).solve(world)
        assert res.solved
        assert res.wrong == 0
        assert world.labels() == {
            "alpha": True,
            "beta": False,
            "gamma": False,
            "delta": True,
        }

    def test_induce_learns_count_rule_from_examples(self) -> None:
        """No stated rule — worked examples alone should carry it."""
        pred_of = ("lens", "spring", "dial")
        examples = [
            Item(0, "e1", ("lens", "spring"), True),
            Item(1, "e2", ("spring", "dial"), True),
            Item(2, "e3", ("lens", "spring", "dial"), False),
            Item(3, "e4", ("dial",), False),
        ]
        items = [
            Item(4, "alpha", ("lens", "spring"), True),
            Item(5, "beta", ("lens", "spring", "dial"), False),
            Item(6, "gamma", ("dial",), False),
            Item(7, "delta", ("lens", "dial"), True),
        ]
        world = RuleGame(None, items, examples)
        res = RuleAgent(seed=2).solve(world)
        assert res.solved
        del pred_of

    def test_wrong_guess_is_supervision_not_death(self) -> None:
        """A wrong guess reveals truth — the agent recovers and solves."""
        items = [
            Item(0, "a", ("x", "y"), True),
            Item(1, "b", ("z",), False),
        ]
        world = RuleGame(None, items)
        agent = RuleAgent(seed=0, epsilon=0.0)
        res = agent.solve(world)
        assert res.solved  # ≤2 guesses per item — always converges

    def test_shared_competence(self) -> None:
        tc = TaskCompetence()
        world = RuleGame(_tessel_predicate(), _tessel_items())
        RuleAgent(task_competence=tc, seed=1).solve(world)
        assert tc.skills  # a verified episode consolidated
        skill = next(iter(tc.skills.values()))
        assert skill.signature.domain == "classification.rule"
        assert any(
            s.family == "feature" for s in skill.steps
        )


class TestClassificationSpec:
    def test_apply_spec_normalizes(self) -> None:
        task = normalize_offered(
            {
                "name": "tessel_game",
                "family": "classification",
                "kind": "tessel",
                "predicate": _tessel_predicate(),
                "items": [
                    {"name": "alpha", "has": ["lens", "spring"]},
                    {"name": "beta", "has": ["lens", "spring", "dial"]},
                    {"name": "gamma", "has": ["dial"]},
                ],
            },
            "x",
        )
        assert task is not None
        assert task["family"] == "classification"
        assert task["kind"] == "tessel"
        assert len(task["items"]) == 3

    def test_induce_spec_requires_labels(self) -> None:
        # Induce mode: unlabeled items have no answer key → rejected.
        assert (
            normalize_offered(
                {
                    "family": "classification",
                    "items": [{"name": "a", "has": ["x"]}],
                },
                "x",
            )
            is None
        )
        task = normalize_offered(
            {
                "family": "classification",
                "items": [{"name": "a", "has": ["x"], "label": True}],
            },
            "x",
        )
        assert task is not None

    def test_contradicted_predicate_rejected(self) -> None:
        """An asserted label that disagrees with the stated predicate
        rejects the spec — the same predicate can't verify itself."""
        assert (
            normalize_offered(
                {
                    "family": "classification",
                    "predicate": {"op": "any_of", "of": ["lens"]},
                    "items": [
                        {
                            "name": "alpha",
                            "has": ["dial"],
                            "label": True,  # asserts a member, pred says no
                        },
                    ],
                },
                "x",
            )
            is None
        )

    def test_malformed_specs_never_crash(self) -> None:
        garbage = [
            {"family": "classification"},
            {"family": "classification", "items": "nope"},
            {"family": "classification", "items": [{"has": "x"}]},
            {"family": "classification", "predicate": {"op": "?"}, "items": []},
            {"family": "classification", "items": [{"name": "a"}]},
            {"family": "classification", "predicate": None, "items": []},
        ]
        for g in garbage:
            assert normalize_offered(g, "x") is None

    def test_normalized_spec_round_trips(self, tmp_path) -> None:
        """A normalized spec written by offer() must re-validate."""
        spec = normalize_offered(
            {
                "name": "rt",
                "family": "classification",
                "predicate": {"op": "any_of", "of": ["lens"]},
                "items": [{"name": "a", "has": ["lens"]}],
            },
            "x",
        )
        assert spec is not None
        practice = SpatialPractice(str(tmp_path))
        path = practice.offer(spec)
        assert path is not None and path.exists()
        offered = practice.offered_tasks()
        assert any(t["name"] == "rt" for t in offered)

    def test_practice_attempt_apply(self, tmp_path) -> None:
        """End to end through SpatialPractice.attempt's task override."""
        from genesis_cognitive.concepts import ConceptNetwork
        from genesis_cognitive.spatial.solver import SpatialReasoner

        practice = SpatialPractice(str(tmp_path))
        reasoner = SpatialReasoner(ConceptNetwork())
        task = normalize_offered(
            {
                "name": "heard_tessel",
                "family": "classification",
                "kind": "tessel",
                "predicate": _tessel_predicate(),
                "items": [
                    {"name": n, "has": list(a)}
                    for n, a in (
                        ("alpha", ("lens", "spring")),
                        ("beta", ("lens", "spring", "dial")),
                        ("gamma", ("dial",)),
                    )
                ],
            },
            "x",
        )
        assert task is not None
        result = practice.attempt(reasoner, task=task)
        assert result is not None
        assert result.solved and result.family == "classification"
        assert result.details["labels"]["alpha"] is True
        assert result.details["labels"]["beta"] is False
        # The file drop + mastery bookkeeping happened.
        assert practice.mastery["heard_tessel"] == 1.0
        # The competence it used is the reasoner's shared store.
        assert reasoner.task_competence.skills


class TestProblemIntake:
    """The outer→inner bridge: described problems → task specs."""

    def test_tessel_compiles_to_apply_spec(self) -> None:
        from genesis_cognitive.cognition.problem_intake import (
            compile_problem,
        )

        spec = compile_problem(
            "A tessel is a machine that has exactly two of these parts: "
            "a lens, a spring, or a dial. Alpha has a lens and a spring. "
            "Beta has a lens, a spring, and a dial. Gamma has only a "
            "dial. Which are tessels?"
        )
        assert spec is not None
        assert spec["family"] == "classification"
        assert spec["kind"] == "tessel"
        assert spec["predicate"] == {
            "op": "exactly",
            "of": ["lens", "spring", "dial"],
            "count": 2,
        }
        items = {i["name"]: set(i["has"]) for i in spec["items"]}
        assert items == {
            "alpha": {"lens", "spring"},
            "beta": {"lens", "spring", "dial"},
            "gamma": {"dial"},
        }
        # And it validates.
        assert normalize_offered(spec, "x") is not None

    def test_label_assertions_attach_as_evidence(self) -> None:
        from genesis_cognitive.cognition.problem_intake import (
            compile_problem,
        )

        spec = compile_problem(
            "Beta is not a wug. A wug is a bird that has a crest and a "
            "tail. Beta has a crest. Lulu has a crest and a tail. "
            "Which are wugs?"
        )
        assert spec is not None
        items = {i["name"]: i for i in spec["items"]}
        assert items["beta"]["label"] is False
        # The assertion agrees with the predicate → valid spec.
        assert normalize_offered(spec, "x") is not None

    def test_contradicted_assertion_rejected_by_normalizer(self) -> None:
        from genesis_cognitive.cognition.problem_intake import (
            compile_problem,
        )

        # User asserts alpha IS a wug, but alpha's parts violate the
        # stated rule — the spec is internally inconsistent.
        spec = compile_problem(
            "Alpha is a wug. A wug is a bird that has a crest and a "
            "tail. Alpha has only a beak. Which are wugs?"
        )
        # May compile (it's a real shape), but must not validate —
        # the stated rule contradicts the asserted label.
        if spec is not None:
            assert normalize_offered(spec, "x") is None

    def test_sequence_compiles(self) -> None:
        from genesis_cognitive.cognition.problem_intake import (
            compile_problem,
        )

        spec = compile_problem("what comes next in a b a b a b?")
        assert spec is not None
        assert spec["family"] == "sequence"
        assert spec["pattern"] == ["a", "b"]
        assert normalize_offered(spec, "x") is not None

    def test_relations_and_quantities_compile(self) -> None:
        from genesis_cognitive.cognition.problem_intake import (
            compile_problem,
        )

        rel = compile_problem(
            "put the star left of the moon and the moon next to the sun"
        )
        assert rel is not None and rel["family"] == "relations"
        assert {g["rel"] for g in rel["goals"]} == {"left_of", "next_to"}
        assert normalize_offered(rel, "x") is not None

        qty = compile_problem("make 9 with groups of 2, 5 and 4")
        assert qty is not None and qty["family"] == "quantities"
        assert qty["target"] == 9
        assert sorted(g["count"] for g in qty["groups"]) == [2, 4, 5]
        assert normalize_offered(qty, "x") is not None

    def test_non_problem_text_returns_none(self) -> None:
        from genesis_cognitive.cognition.problem_intake import (
            compile_problem,
        )

        for text in (
            "hello genesis, how are you today?",
            "the cat is left of the dog",  # scene description, no cue
            "tell me about yourself",
            "what is your favorite color?",
            "",
        ):
            assert compile_problem(text) is None

    def test_interpretation_is_semantic_not_surface(self) -> None:
        """The real entry point takes a ComprehensionResult — intake
        never sees raw text; it reads propositions, roles, negation."""
        from genesis_cognitive.cognition.problem_intake import (
            interpret_problem,
        )
        from genesis_cognitive.language.comprehension import (
            ComprehensionEngine,
        )

        result = ComprehensionEngine().comprehend(
            "A tessel is a machine that has exactly two of these "
            "parts: a lens, a spring, or a dial. Alpha has a lens "
            "and a spring. Which are tessels?"
        )
        spec = interpret_problem(result)
        assert spec is not None
        assert spec["family"] == "classification"
        assert spec["predicate"]["op"] == "exactly"

    def test_rephrasing_variants_compile(self) -> None:
        """Semantic robustness: same task, different surface forms."""
        from genesis_cognitive.cognition.problem_intake import (
            compile_problem,
        )

        # Gerund reduced relative ("object containing...").
        spec = compile_problem(
            "A quark is any object containing at least two of: "
            "a spin, a charge, a color. Alpha contains a spin and "
            "a charge. Beta contains all three. Which are quarks?"
        )
        assert spec is not None
        assert spec["predicate"]["op"] == "at_least"
        items = {i["name"]: set(i["has"]) for i in spec["items"]}
        assert items["beta"] == {"spin", "charge", "color"}

        # Plural generic subject ("Tessels are machines that...").
        spec = compile_problem(
            "Tessels are machines that have exactly two of these "
            "parts: a lens, a spring, or a dial. Alpha has a lens "
            "and a spring. Which ones are tessels?"
        )
        assert spec is not None and spec["kind"] == "tessel"

        # Quantity variants: "reach N using groups of...".
        spec = compile_problem("reach 12 using groups of 3, 4 and 5")
        assert spec is not None and spec["target"] == 12
        assert sorted(g["count"] for g in spec["groups"]) == [3, 4, 5]
        spec = compile_problem("build 8 using groups of 3 and 5")
        assert spec is not None and spec["target"] == 8

        # Declarative arrangement ("the star should be left of...").
        spec = compile_problem(
            "the star should be left of the moon and the moon "
            "next to the sun"
        )
        assert spec is not None and spec["family"] == "relations"
        assert {g["rel"] for g in spec["goals"]} == {
            "left_of", "next_to"
        }

    def test_heard_problem_solves_through_practice(self, tmp_path) -> None:
        """The whole bridge: words → spec → drop-box → attempt → answer."""
        from genesis_cognitive.cognition.problem_intake import (
            compile_problem,
        )
        from genesis_cognitive.concepts import ConceptNetwork
        from genesis_cognitive.spatial.solver import SpatialReasoner

        practice = SpatialPractice(str(tmp_path))
        reasoner = SpatialReasoner(ConceptNetwork())
        spec = compile_problem(
            "A tessel is a machine that has exactly two of these parts: "
            "a lens, a spring, or a dial. Alpha has a lens and a spring. "
            "Beta has a lens, a spring, and a dial. Which are tessels?"
        )
        task = normalize_offered(spec, "heard")
        assert task is not None
        assert practice.offer(task) is not None
        result = practice.attempt(reasoner, task=task)
        assert result is not None and result.solved
        assert result.details["labels"] == {"alpha": True, "beta": False}
        # The spec persists as a file — the inner world holds it.
        offered_dir = practice.offered_dir()
        assert (offered_dir / f"{task['name']}.json").exists()
        raw = json.loads((offered_dir / f"{task['name']}.json").read_text())
        assert raw["family"] == "classification"


class TestEngineRoute:
    """The cognition-level path: a described problem goes through
    deliberation, lands in the drop-box, gets worked, and returns as
    a Thought — the outer→inner→outer loop."""

    def _engine(self, tmp_path):
        import os
        import sys
        from types import SimpleNamespace

        sys.path.insert(
            0,
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        )
        from genesis_client import GenesisClient
        from genesis_cognitive.cognition.engine import CognitionEngine
        from genesis_cognitive.concepts import ConceptNetwork
        from genesis_cognitive.language import GenerativeEngine
        from genesis_cognitive.memory import MemoryEngine
        from genesis_cognitive.self import SelfModel

        data_dir = str(tmp_path / "mind")
        os.makedirs(data_dir, exist_ok=True)
        client = GenesisClient(os.path.join(data_dir, "genesis.sock"))
        self_model = SelfModel()
        network = ConceptNetwork()
        memory = MemoryEngine(
            client, network=network, get_emotion=lambda: None
        )
        language = GenerativeEngine(self_model, seed=1, network=network)
        eng = CognitionEngine(
            client=client,
            self_model=self_model,
            memory=memory,
            language=language,
            network=network,
            data_dir=data_dir,
        )
        return eng, SimpleNamespace

    def test_deliberate_problem_offer(self, tmp_path) -> None:
        from types import SimpleNamespace

        eng, _ = self._engine(tmp_path)
        practice = SpatialPractice(str(tmp_path / "practice"))
        eng.set_problem_intake(practice, None, None)
        perception = SimpleNamespace(
            raw_text=(
                "A tessel is a machine that has exactly two of these "
                "parts: a lens, a spring, or a dial. Alpha has a lens "
                "and a spring. Beta has a lens, a spring, and a dial. "
                "Which are tessels?"
            ),
            topics=["tessel"],
        )
        emotion = SimpleNamespace(label="curious")
        thought = eng._deliberate_problem_offer(perception, emotion)
        assert thought is not None
        md = thought.metadata["problem_result"]
        assert md["solved"] and md["family"] == "classification"
        assert md["yes"] == ["alpha"]
        assert md["kind"] == "tessel"
        # It landed in the inner world — a real file in the drop-box.
        offered = practice.offered_dir()
        assert list(offered.glob("heard_*.json"))
        # And it went through the shared competence store.
        assert eng.task_competence.skills

    def test_offer_runs_on_comprehension_result(self, tmp_path) -> None:
        """The plumbed path: intake interprets the ComprehensionResult
        think() already computed — it does not re-parse raw text."""
        from types import SimpleNamespace

        from genesis_cognitive.language.comprehension import (
            ComprehensionEngine,
        )

        eng, _ = self._engine(tmp_path)
        practice = SpatialPractice(str(tmp_path / "practice"))
        eng.set_problem_intake(practice, None, None)
        text = (
            "A wug is a bird that has a crest and a tail. Lulu has "
            "a crest and a tail. Beta has only a crest. Which are "
            "wugs?"
        )
        comp = ComprehensionEngine().comprehend(text)
        perception = SimpleNamespace(raw_text=text, topics=["wug"])
        emotion = SimpleNamespace(label="curious")
        thought = eng._deliberate_problem_offer(
            perception, emotion, comp
        )
        assert thought is not None
        md = thought.metadata["problem_result"]
        assert md["solved"] and md["family"] == "classification"
        assert md["yes"] == ["lulu"]
        assert md["no"] == ["beta"]

    def test_plain_conversation_passes_through(self, tmp_path) -> None:
        from types import SimpleNamespace

        eng, _ = self._engine(tmp_path)
        practice = SpatialPractice(str(tmp_path / "practice"))
        eng.set_problem_intake(practice, None, None)
        perception = SimpleNamespace(
            raw_text="tell me about your day", topics=["day"]
        )
        emotion = SimpleNamespace(label="neutral")
        assert (
            eng._deliberate_problem_offer(perception, emotion) is None
        )
        # Nothing was offered — the drop-box stays empty.
        assert not list(practice.offered_dir().glob("*.json"))

    def test_unwired_engine_ignores_problems(self, tmp_path) -> None:
        from types import SimpleNamespace

        eng, _ = self._engine(tmp_path)
        perception = SimpleNamespace(
            raw_text="make 5 with groups of 2 and 3", topics=[]
        )
        emotion = SimpleNamespace(label="neutral")
        # No practice wired → the route declines rather than crash.
        assert (
            eng._deliberate_problem_offer(perception, emotion) is None
        )

    def test_inconsistent_problem_reports_honestly(
        self, tmp_path
    ) -> None:
        from types import SimpleNamespace

        eng, _ = self._engine(tmp_path)
        practice = SpatialPractice(str(tmp_path / "practice"))
        eng.set_problem_intake(practice, None, None)
        perception = SimpleNamespace(
            raw_text=(
                "Alpha is a wug. A wug is a bird that has a crest and "
                "a tail. Alpha has only a beak. Which are wugs?"
            ),
            topics=["wug"],
        )
        emotion = SimpleNamespace(label="neutral")
        thought = eng._deliberate_problem_offer(perception, emotion)
        if thought is not None:
            # Either declined entirely (compile found nothing solid)
            # or reported invalid — never a fabricated answer.
            assert thought.metadata["problem_result"]["invalid"]
            assert not thought.metadata["problem_result"]["solved"]


class TestSorterIntake:
    """The sorter span of the intake bridge: a described shape
    sorter compiles to slots/blocks/lid and validates."""

    def test_box_with_holes_compiles(self) -> None:
        from genesis_cognitive.cognition.problem_intake import (
            compile_problem,
        )

        spec = compile_problem(
            "the box has two holes: a round hole and a square hole. "
            "there is a red round block and a blue square block. "
            "put each block in its hole."
        )
        assert spec is not None
        assert spec["family"] == "sorter"
        accepts = [set(s["accepts"]) for s in spec["slots"]]
        assert {"round"} in accepts and {"square"} in accepts
        attrs = [set(b["attrs"]) for b in spec["blocks"]]
        assert {"red", "round"} in attrs
        assert {"blue", "square"} in attrs
        assert normalize_offered(spec, "x") is not None

    def test_takes_and_fits_frames_compile(self) -> None:
        from genesis_cognitive.cognition.problem_intake import (
            compile_problem,
        )

        spec = compile_problem(
            "the star hole takes a small star and the moon hole "
            "takes a big moon. put each shape in its hole."
        )
        assert spec is not None and spec["family"] == "sorter"
        assert normalize_offered(spec, "x") is not None

        spec = compile_problem(
            "a small star fits the star hole. "
            "a big moon fits the moon hole. sort them."
        )
        assert spec is not None and spec["family"] == "sorter"
        assert normalize_offered(spec, "x") is not None

    def test_gapped_imperative_compiles(self) -> None:
        from genesis_cognitive.cognition.problem_intake import (
            compile_problem,
        )

        spec = compile_problem(
            "put the red star in the star hole "
            "and the blue moon in the moon hole"
        )
        assert spec is not None and spec["family"] == "sorter"
        assert len(spec["slots"]) == 2
        assert len(spec["blocks"]) == 2
        assert normalize_offered(spec, "x") is not None

        # A mismatched instruction is honestly rejected: "put the
        # red star in the SQUARE hole" states a placement the
        # conjunctive criterion ({square}) can't satisfy.
        bad = compile_problem(
            "put the red star in the square hole "
            "and the blue moon in the round hole"
        )
        if bad is not None:
            assert normalize_offered(bad, "x") is None

    def test_scene_without_task_signal_returns_none(self) -> None:
        from genesis_cognitive.cognition.problem_intake import (
            compile_problem,
        )

        # Slots and pieces described, but nothing asks her to sort
        # and no acceptance claim is made — description, not task.
        assert (
            compile_problem(
                "the shelf is tall. the round hole is on the box."
            )
            is None
        )
        assert compile_problem("sort them") is None

    def test_unsolvable_description_does_not_validate(self) -> None:
        from genesis_cognitive.cognition.problem_intake import (
            compile_problem,
        )

        # Two constrained slots, but the only declared piece
        # fills one — the square hole stays unfilled, and bipartite
        # matching fails.
        spec = compile_problem(
            "the box has a round hole and a square hole. "
            "there is a round peg. sort them."
        )
        if spec is not None:
            assert normalize_offered(spec, "x") is None

    def test_heard_sorter_solves_through_practice(
        self, tmp_path
    ) -> None:
        """Words → spec → drop-box → attempt → solved."""
        from genesis_cognitive.cognition.problem_intake import (
            compile_problem,
        )
        from genesis_cognitive.concepts import ConceptNetwork
        from genesis_cognitive.spatial.solver import SpatialReasoner

        practice = SpatialPractice(str(tmp_path))
        reasoner = SpatialReasoner(ConceptNetwork())
        spec = compile_problem(
            "the box has two holes: a round hole and a square hole. "
            "there is a red round block and a blue square block. "
            "put each block in its hole."
        )
        task = normalize_offered(spec, "heard")
        assert task is not None
        assert practice.offer(task) is not None
        result = practice.attempt(reasoner, task=task)
        assert result is not None and result.solved
