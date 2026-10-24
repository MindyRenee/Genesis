"""Tests for the symbolic-math faculty and the research workbench.

The pair is the first rung of mathematical capability: a CAS the
mind can compute with (``parietal_lobe/math/symbolic.py``) and a
persistent bench where open problems, claims, and failed attempts
accumulate (``frontal_lobe/research.py``).

The invariant under test is provenance discipline: cognition may
record conjectures, readings, and empirical support, but the
terminal verdicts — "verified" and "refuted" — are written only by
``verify``, which consults the oracle itself. A workbench that let
callers assert "verified" directly would be a diary, not a proof
substrate.
"""

from __future__ import annotations

import random

import pytest
from frontal_lobe import research as research_module
from frontal_lobe.research import ResearchWorkbench
from frontal_lobe.tools.framework import ToolRegistry, get_tools
from frontal_lobe.volition import Urge
from parietal_lobe.math.symbolic import (
    check_equality,
    compute,
    mutate_identity,
    split_equality,
)

from infrastructure.config import VolitionConfig

sympy = pytest.importorskip("sympy")

# ─── compute: algebra and calculus ───────────────────────────────


def test_simplify_reduces_rational_expression():
    result = compute("simplify", "(x**2 - 1)/(x - 1)")
    assert result.ok
    assert result.output == "x + 1"


def test_expand_and_factor_are_inverses():
    expanded = compute("expand", "(x + 1)**3")
    assert expanded.ok
    factored = compute("factor", expanded.output)
    assert factored.ok
    assert factored.output == "(x + 1)**3"


def test_diff_and_integrate_are_inverses():
    derivative = compute("diff", "x**3")
    assert derivative.ok
    assert derivative.output == "3*x**2"
    integral = compute("integrate", derivative.output)
    assert integral.ok
    assert integral.output == "x**3"


def test_diff_honors_order():
    result = compute("diff", "x**3", order=2)
    assert result.ok
    assert result.output == "6*x"


def test_definite_integral_needs_both_bounds():
    ok = compute("integrate", "3*x**2", lower="0", upper="1")
    assert ok.ok
    assert ok.output == "1"
    bad = compute("integrate", "3*x**2", lower="0")
    assert not bad.ok


def test_solve_bare_expression_and_equation():
    bare = compute("solve", "x**2 - 2")
    assert bare.ok
    assert "sqrt(2)" in bare.data["solutions"]
    equation = compute("solve", "x**2 = 4")
    assert equation.ok
    assert set(equation.data["solutions"]) == {"-2", "2"}


def test_limit_and_series():
    assert compute("limit", "sin(x)/x", point="0").output == "1"
    series = compute("series", "sin(x)", point="0", order=5)
    assert series.ok
    assert "x - x**3/6" in series.output


def test_evalf_gives_numeric_value():
    result = compute("evalf", "pi")
    assert result.ok
    assert result.output.startswith("3.14159")


def test_matrix_ops():
    assert compute("det", "[[1, 2], [3, 4]]").output == "-2"
    assert compute("trace", "[[1, 2], [3, 4]]").output == "5"
    eig = compute("eigenvals", "[[2, 0], [0, 3]]")
    assert eig.ok
    assert "2" in eig.output and "3" in eig.output


def test_natural_notation_parses():
    result = compute("simplify", "2x + 3x")
    assert result.ok
    assert result.output == "5*x"


def test_unknown_operation_and_bad_input_fail_cleanly():
    assert not compute("frobnicate", "x").ok
    assert not compute("simplify", "x +* 2").ok


# ─── check_equality: the oracle ──────────────────────────────────


def test_oracle_verifies_a_true_equality():
    verdict = check_equality("(x + 1)**2", "x**2 + 2*x + 1")
    assert verdict.status == "verified"


def test_oracle_refutes_with_a_witness():
    verdict = check_equality("x**2", "2*x")
    assert verdict.status == "refuted"
    assert verdict.witness is not None
    # x=2 satisfies it, so the witness must be a different point.
    assert verdict.witness["substitution"] != {"x": "2"}


def test_oracle_refutes_everywhere_when_the_gap_is_constant():
    verdict = check_equality("x", "x + 1")
    assert verdict.status == "refuted"
    # An empty substitution means no assignment can satisfy it.
    assert verdict.witness == {
        "substitution": {},
        "difference": verdict.witness["difference"],
    }


def test_oracle_rejects_unparseable_input():
    with pytest.raises(ValueError):
        check_equality("x +* 2", "x")


# ─── Equational parsing and mutation ─────────────────────────────


def test_split_equality_only_accepts_bare_equals():
    assert split_equality("x + 1 = 2*x - 1") == ("x + 1", "2*x - 1")
    assert split_equality("x <= 2*x") is None
    assert split_equality("x == 2*x") is None
    assert split_equality("x != 2*x") is None
    assert split_equality("the mass gap is positive") is None


def test_mutate_identity_shape_and_determinism():
    rng = random.Random(7)
    candidate = mutate_identity("(x + 1)**2", "x**2 + 2*x + 1", rng)
    assert candidate is not None
    assert {"lhs", "rhs", "method", "sound"} <= candidate.keys()
    same = mutate_identity("(x + 1)**2", "x**2 + 2*x + 1", random.Random(7))
    assert same == candidate


def test_sound_mutations_are_never_refuted():
    # A sound move applied to a true identity cannot produce a
    # refutable claim — the oracle may confirm or fail to decide,
    # but it must never find a counterexample to valid inference.
    rng = random.Random(11)
    for _ in range(20):
        candidate = mutate_identity("x + x", "2*x", rng)
        if candidate is None or not candidate["sound"]:
            continue
        verdict = check_equality(candidate["lhs"], candidate["rhs"])
        assert verdict.status != "refuted", candidate


# ─── Work sessions ───────────────────────────────────────────────


def test_work_on_empty_bench_produces_nothing(tmp_path):
    bench = ResearchWorkbench(tmp_path)
    assert bench.work()["thread"] is None


def test_work_judges_equational_conjectures(tmp_path):
    bench = ResearchWorkbench(tmp_path)
    thread = bench.open_problem("test problem")
    bench.record_conjecture(thread.id, "x + x = 2*x")
    bench.record_conjecture(thread.id, "unformalized prose, not an equation")
    report = bench.work(thread.id)
    assert report["verified"] == ["c1"]
    # The prose claim is untouched — the oracle cannot reach it.
    assert bench.thread(thread.id).claims[1].provenance == "conjectured"


def test_work_proposes_a_mutation_when_nothing_is_open(tmp_path):
    bench = ResearchWorkbench(tmp_path)
    thread = bench.open_problem("test problem")
    claim = bench.record_conjecture(thread.id, "x + x = 2*x")
    bench.verify(thread.id, claim.id, "x + x", "2*x")
    proposed = []
    for _ in range(5):
        report = bench.work(thread.id)
        if report["proposed"]:
            proposed = report["proposed"]
            break
    assert proposed
    new_claim = next(c for c in thread.claims if c.id == proposed[0])
    # Whatever the mutation was, it went through the oracle — its
    # provenance is terminal, never asserted.
    assert new_claim.provenance in {"verified", "refuted"}


def test_work_skips_paused_and_resolved_threads(tmp_path):
    bench = ResearchWorkbench(tmp_path)
    thread = bench.open_problem("test problem")
    bench.pause(thread.id)
    assert bench.work()["thread"] is None


# ─── The urge is object-directed ─────────────────────────────────


def test_research_urge_is_registered():
    urges = {u.name: u for u in VolitionConfig().urges}
    assert "research" in urges
    assert urges["research"].growth == 0.0
    assert set(urges["research"].stimuli) == {"open_problems", "curiosity"}


def test_research_urge_cannot_form_without_an_object():
    # growth=0 means desire needs an object: an empty bench produces
    # no stimulus and no pressure, however long she sits with it.
    urge = Urge(
        name="research",
        threshold=0.70,
        growth=0.0,
        decay=0.0004,
        cooldown=900.0,
        stimuli={"open_problems": 0.05, "curiosity": 0.02},
    )
    urge.tick({}, 300.0)
    assert urge.value == 0.0
    urge.tick({"open_problems": 0.0, "curiosity": 0.0}, 300.0)
    assert urge.value == 0.0
    urge.tick({"open_problems": 1.0, "curiosity": 0.3}, 20.0)
    assert urge.is_ready()


def test_curiosity_alone_does_not_sustain_the_urge():
    urge = Urge(
        name="research",
        threshold=0.70,
        growth=0.0,
        decay=0.0004,
        cooldown=900.0,
        stimuli={"open_problems": 0.05, "curiosity": 0.02},
    )
    urge.tick({"curiosity": 0.5}, 60.0)
    assert not urge.is_ready()


# ─── ResearchWorkbench ───────────────────────────────────────────


def test_open_problem_persists_and_dedupes(tmp_path):
    bench = ResearchWorkbench(tmp_path)
    thread = bench.open_problem("Does the mass gap close?")
    again = bench.open_problem("  does the MASS GAP close? ")
    assert again.id == thread.id

    reloaded = ResearchWorkbench(tmp_path)
    assert reloaded.thread(thread.id) is not None
    assert reloaded.thread(thread.id).statement == thread.statement


def test_claims_are_born_unverified(tmp_path):
    bench = ResearchWorkbench(tmp_path)
    thread = bench.open_problem("test problem")
    claim = bench.record_conjecture(thread.id, "x + x = 2x")
    assert claim.provenance == "conjectured"
    reading = bench.record_reading(thread.id, "heat equation", source="arxiv")
    assert reading.provenance == "read"


def test_verified_provenance_only_comes_through_the_oracle(tmp_path):
    bench = ResearchWorkbench(tmp_path)
    thread = bench.open_problem("test problem")
    claim = bench.record_conjecture(thread.id, "expansion identity")
    result = bench.verify(thread.id, claim.id, "(x + 1)**2", "x**2 + 2*x + 1")
    assert result["verdict"].status == "verified"
    assert result["claim"].provenance == "verified"
    assert result["claim"].verification["lhs"] == "(x + 1)**2"


def test_refuted_claims_carry_the_counterexample(tmp_path):
    bench = ResearchWorkbench(tmp_path)
    thread = bench.open_problem("test problem")
    claim = bench.record_conjecture(thread.id, "wrong identity")
    result = bench.verify(thread.id, claim.id, "x", "x + 1")
    assert result["claim"].provenance == "refuted"
    assert any("counterexample" in e for e in result["claim"].evidence)


def test_terminal_claims_are_not_rejudged_or_demoted(tmp_path):
    bench = ResearchWorkbench(tmp_path)
    thread = bench.open_problem("test problem")
    claim = bench.record_conjecture(thread.id, "identity")
    bench.verify(thread.id, claim.id, "x + x", "2*x")
    again = bench.verify(thread.id, claim.id, "x + x", "x + x + x")
    assert again["claim"].provenance == "verified"
    with pytest.raises(ValueError):
        bench.mark_empirical(thread.id, claim.id, "more evidence")


def test_empirical_support_is_not_a_verdict(tmp_path):
    bench = ResearchWorkbench(tmp_path)
    thread = bench.open_problem("test problem")
    claim = bench.record_conjecture(thread.id, "growth bound")
    marked = bench.mark_empirical(thread.id, claim.id, "holds for n<1000")
    assert marked.provenance == "empirical"


def test_resolve_requires_a_terminal_verdict(tmp_path):
    bench = ResearchWorkbench(tmp_path)
    thread = bench.open_problem("test problem")
    claim = bench.record_conjecture(thread.id, "identity")
    with pytest.raises(ValueError):
        bench.resolve(thread.id)
    bench.verify(thread.id, claim.id, "x + x", "2*x")
    assert bench.resolve(thread.id).status == "resolved"
    with pytest.raises(ValueError):
        bench.record_conjecture(thread.id, "late claim")


def test_attempts_accumulate_in_the_graveyard(tmp_path):
    bench = ResearchWorkbench(tmp_path)
    thread = bench.open_problem("test problem")
    bench.record_attempt(thread.id, "induction on n", note="base case fails at n=3")
    bench.record_attempt(thread.id, "direct expansion", outcome="partial")
    reloaded = ResearchWorkbench(tmp_path)
    attempts = reloaded.thread(thread.id).attempts
    assert len(attempts) == 2
    assert attempts[0].outcome == "failed"
    assert attempts[1].outcome == "partial"


def test_status_summarizes_claims_by_provenance(tmp_path):
    bench = ResearchWorkbench(tmp_path)
    thread = bench.open_problem("test problem")
    claim = bench.record_conjecture(thread.id, "identity")
    bench.verify(thread.id, claim.id, "x + x", "2*x")
    summary = bench.status(thread.id)
    assert summary["claims"]["verified"] == 1
    whole = bench.status()
    assert whole["open"] == 0 or whole["open"] == 1
    assert whole["threads"]


def test_missing_thread_and_claim_raise(tmp_path):
    bench = ResearchWorkbench(tmp_path)
    with pytest.raises(KeyError):
        bench.record_conjecture("nope", "x")
    thread = bench.open_problem("test problem")
    with pytest.raises(KeyError):
        bench.verify(thread.id, "c99", "x", "x")


# ─── Tool registration ───────────────────────────────────────────


def test_registry_exposes_the_new_tools():
    names = {t["name"] for t in ToolRegistry().list_tools()}
    assert {"symbolic_math", "verify_equality", "research"} <= names


def test_symbolic_math_tool_runs():
    result = get_tools().run("symbolic_math", operation="simplify", expression="(x**2 - 1)/(x - 1)")
    assert result.success
    assert result.output == "x + 1"


def test_verify_equality_tool_runs():
    result = get_tools().run("verify_equality", lhs="x + x", rhs="2*x")
    assert result.success
    assert result.data["status"] == "verified"


def test_research_tool_roundtrip(tmp_path, monkeypatch):
    bench = ResearchWorkbench(tmp_path)
    monkeypatch.setattr(research_module, "get_workbench", lambda: bench)
    tools = get_tools()

    opened = tools.run("research", action="open", statement="test problem")
    assert opened.success
    thread_id = opened.data["thread"]["id"]

    c = tools.run("research", action="conjecture", thread=thread_id, statement="x + x = 2x")
    assert c.data["claim"]["provenance"] == "conjectured"

    v = tools.run(
        "research",
        action="verify",
        thread=thread_id,
        claim="c1",
        lhs="x + x",
        rhs="2*x",
    )
    assert v.data["verdict"]["status"] == "verified"

    assert tools.run("research", action="resolve", thread=thread_id).success
    status = tools.run("research", action="status")
    assert status.data["status"]["resolved"] == 1


def test_research_tool_reports_errors(tmp_path, monkeypatch):
    bench = ResearchWorkbench(tmp_path)
    monkeypatch.setattr(research_module, "get_workbench", lambda: bench)
    tools = get_tools()
    assert not tools.run("research", action="bogus").success
    assert not tools.run("research", action="conjecture", thread="missing", statement="x").success


# ─── A wider oracle: inequalities and bounded universals ─────────


def test_inequality_verified_symbolically():
    from parietal_lobe.math.symbolic import check_inequality

    v = check_inequality("x**2", "0", ">=")
    assert v.status == "verified"


def test_inequality_refuted_with_witness():
    from parietal_lobe.math.symbolic import check_inequality

    v = check_inequality("x", "x + 1", ">=")
    assert v.status == "refuted"
    assert v.witness is not None


def test_inequality_undecided_is_honest():
    from parietal_lobe.math.symbolic import check_inequality

    # x > |x| has no real solution region sympy will certify either
    # way; the oracle must say so rather than round to a verdict.
    v = check_inequality("x", "Abs(x)", ">")
    assert v.status in {"refuted", "undecided"}


def test_bounded_equality_pass_is_evidence_not_proof():
    from parietal_lobe.math.symbolic import check_bounded_equality

    # True everywhere, but the bounded check only covers a range —
    # it must NOT write "verified".
    v = check_bounded_equality("n**2 + n", "n * (n + 1)", "n", 1, 20)
    assert v.status == "undecided"
    assert "[1,20]" in v.detail


def test_bounded_equality_refutes_with_counterexample():
    from parietal_lobe.math.symbolic import check_bounded_equality

    # False already at n=1 — the witness is the failing integer.
    v = check_bounded_equality("(n + 1)**2", "n**2 + 1", "n", 1, 10)
    assert v.status == "refuted"
    assert v.witness["substitution"] == {"n": "1"}


def test_split_claim_parses_relations_and_bounds():
    from parietal_lobe.math.symbolic import split_claim

    assert split_claim("x + 1 = 1 + x")["op"] == "="
    assert split_claim("x**2 >= 0")["op"] == ">="
    bounded = split_claim("n*n = n**2 for n <= 20")
    assert bounded["var"] == "n" and bounded["hi"] == 20
    bounded2 = split_claim("for all n <= 10: n + 1 = 1 + n")
    assert bounded2["var"] == "n" and bounded2["op"] == "="
    assert split_claim("a != b") is None


def test_work_judges_inequality_and_bounded_claims(tmp_path):
    bench = ResearchWorkbench(tmp_path)
    thread = bench.open_problem("relations")
    bench.record_conjecture(thread.id, "x**2 >= 0")
    bench.record_conjecture(thread.id, "(n + 1)**2 = n**2 + 1 for n <= 10")
    bench.record_conjecture(thread.id, "n*n = n**2 for n <= 20")
    report = bench.work(thread.id)
    claims = {c.statement: c for c in thread.claims}
    assert claims["x**2 >= 0"].provenance == "verified"
    assert claims["(n + 1)**2 = n**2 + 1 for n <= 10"].provenance == "refuted"
    # The true bounded claim stays unjudged — evidence, not verdict.
    bounded = claims["n*n = n**2 for n <= 20"]
    assert bounded.provenance == "conjectured"
    assert "holds at every integer" in bounded.evidence[-1]
    assert set(report["verified"]) | set(report["refuted"])


def test_workbench_digest_anchors_the_ledger(tmp_path):
    import hashlib

    bench = ResearchWorkbench(tmp_path)
    assert bench.digest() == ""  # nothing saved yet — nothing to anchor
    thread = bench.open_problem("anchor test")
    bench.record_conjecture(thread.id, "x + x = 2*x")
    bench.verify(thread.id, "c1", "x + x", "2*x")
    expected = hashlib.sha256((tmp_path / "research" / "workbench.json").read_bytes()).hexdigest()[
        :10
    ]
    assert bench.digest() == expected
    # Any later mutation moves the anchor — history can't be
    # silently rewritten under a published digest.
    bench.record_conjecture(thread.id, "x*x = x**2")
    assert bench.digest() != expected
